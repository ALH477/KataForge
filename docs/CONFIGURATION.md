# Configuration System

## Structure
- **Base configuration**: `config/base.yaml` (generic settings)
- **Hardware profiles**: `config/profiles/*.yaml` (GPU-specific overrides)
- **LLM configuration**: `config/llm.yaml` (language model settings)

## Profile Activation
Set `DOJO_PROFILE` environment variable:

```bash
# Use AMD ROCm profile
export DOJO_PROFILE=framework16

# Or via command line
DOJO_PROFILE=cuda dojo-manager train
```

Default profile is `framework16`.

## Adding New Profiles
1. Create `config/profiles/mygpu.yaml`
2. Add parameters like:
```yaml
device: rocm
power_limit: 200
precision: fp16
```
3. Test with `DOJO_PROFILE=mygpu`

## GPU Validation
Profiles include hardware requirements:
```yaml
validation:
  cpu_requirement: "AMD Ryzen 7+"  # Example
```

System will warn if mismatch detected.
