# Route-isolated quality presets implementation plan

1. Add failing schema, performance and frontend-source tests for the three new labels, stable keys, route isolation, and sampler/LoRA contracts.
2. Add schema mappings and route lists while preserving old keys and saved workflow normalization.
3. Add backend preset values, v4/Ref2VA model constants, explicit LoRA strengths, scheduler and acceleration labels.
4. Add frontend route-filtered labels and Chinese hints without changing the Director panel geometry.
5. Update API and Chinese usage documentation with the public labels and route behavior.
6. Run the full pytest suite, compile checks and diff whitespace check, then commit and push `main`.

