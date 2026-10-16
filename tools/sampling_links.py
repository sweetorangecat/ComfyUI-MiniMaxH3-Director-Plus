"""Keep H3 sigma schedules attached to the model used by their guider."""

SAMPLERS = {"SamplerCustomAdvanced", "MiniMaxH3TwoStageSampler", "MiniMaxH3MemoryAwareSampler"}


class ScheduleLinkError(ValueError):
    pass


def _value(raw):
    if isinstance(raw, dict):
        if "value" in raw:
            return raw["value"]
        return [raw.get(k) for k in ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")]
    return raw


def scope_scheduler_mismatches(nodes, links):
    node_map = {node.get("id"): node for node in nodes}
    link_map = {_value(raw)[0]: _value(raw) for raw in links or []}

    def incoming(node, name):
        socket = next((item for item in node.get("inputs", []) if item.get("name") == name), {})
        return link_map.get(socket.get("link"))

    mismatches = []
    for sampler in nodes:
        if sampler.get("type") not in SAMPLERS:
            continue
        sigma_link, guider_link = incoming(sampler, "sigmas"), incoming(sampler, "guider")
        if sigma_link is None or guider_link is None:
            continue
        scheduler, guider = node_map.get(sigma_link[1], {}), node_map.get(guider_link[1], {})
        if scheduler.get("type") != "MiniMaxH3SchedulerRouter":
            continue
        actual, expected = incoming(scheduler, "model"), incoming(guider, "model")
        if actual is None or expected is None or actual[1:3] == expected[1:3]:
            continue
        mismatches.append({
            "sampler_id": sampler["id"], "scheduler_id": scheduler["id"],
            "model_link_id": actual[0],
            "old_source": actual[1:3], "new_source": expected[1:3],
        })
    return mismatches


def _scopes(workflow):
    yield "main", workflow
    for scope in workflow.get("definitions", {}).get("subgraphs", []) or []:
        yield scope.get("name") or str(scope.get("id")), scope


def align_h3_scheduler_models(workflow):
    """Rewire existing model links, preserving all controls and other connections.

    Validate every proposed repair before changing anything: one scheduler cannot
    serve guiders with different model outputs without being duplicated explicitly.
    """
    plans = []
    for name, scope in _scopes(workflow):
        expected_by_scheduler = {}
        # Include already aligned consumers when detecting shared-scheduler conflicts.
        nodes, links = scope.get("nodes", []), scope.get("links", [])
        node_map = {node.get("id"): node for node in nodes}
        link_map = {_value(raw)[0]: _value(raw) for raw in links}
        for sampler in nodes:
            if sampler.get("type") not in SAMPLERS:
                continue
            sockets = {s.get("name"): link_map.get(s.get("link")) for s in sampler.get("inputs", [])}
            sigma_link, guider_link = sockets.get("sigmas"), sockets.get("guider")
            if sigma_link is None or guider_link is None:
                continue
            scheduler, guider = node_map.get(sigma_link[1], {}), node_map.get(guider_link[1], {})
            if scheduler.get("type") != "MiniMaxH3SchedulerRouter":
                continue
            model_socket = next((s for s in guider.get("inputs", []) if s.get("name") == "model"), {})
            model_link = link_map.get(model_socket.get("link"))
            if model_link is None:
                continue
            expected = tuple(model_link[1:3])
            previous = expected_by_scheduler.setdefault(scheduler["id"], expected)
            if previous != expected:
                raise ScheduleLinkError(f"{name}: scheduler {scheduler['id']} is shared by different guider models")
        seen = set()
        for change in scope_scheduler_mismatches(nodes, links):
            if change["model_link_id"] not in seen:
                plans.append((name, scope, change))
                seen.add(change["model_link_id"])

    changes = []
    for name, scope, change in plans:
        link_id = change["model_link_id"]
        raw = next(raw for raw in scope["links"] if _value(raw)[0] == link_id)
        if isinstance(raw, dict) and "value" not in raw:
            raw["origin_id"], raw["origin_slot"] = change["new_source"]
        else:
            record = raw["value"] if isinstance(raw, dict) else raw
            record[1:3] = change["new_source"]
        node_map = {node.get("id"): node for node in scope.get("nodes", [])}
        for source, adding in ((change["old_source"], False), (change["new_source"], True)):
            node_id, slot = source
            if node_id == -10:
                sockets = scope.get("inputs", [])
                socket = sockets[slot] if 0 <= slot < len(sockets) else None
                field = "linkIds"
            else:
                sockets = node_map.get(node_id, {}).get("outputs", [])
                socket = sockets[slot] if 0 <= slot < len(sockets) else None
                field = "links"
            if socket is None:
                continue
            ids = list(socket.get(field) or [])
            if adding and link_id not in ids:
                ids.append(link_id)
            elif not adding:
                ids = [value for value in ids if value != link_id]
            socket[field] = ids if field == "linkIds" else ids or None
        changes.append({"scope": name, **change})
    return changes
