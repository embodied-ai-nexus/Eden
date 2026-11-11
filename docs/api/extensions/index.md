# Extensions

Extensions are optional subsystems built on top of the core environment: real-robot deployment, the policy zoo and
remote-policy clients, motion retargeting, system identification, motion planning and optimization, grasp sampling,
pose estimation, scripted demonstrations, teleoperation and visualization. Each lives in a package under
`eden.extensions`, keeps its options class in `eden.options.extensions.<name>`, and imports its heavy backend
lazily, so an extension's optional extra is needed only when you use it.

```python
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY
from eden.options.extensions.deployment import DeploymentOptions
```

| Extension | Page | CLI | Guide |
|---|---|---|---|
| Deployment | [Deployment](deployment.md) | `eden deploy` | [Deployment](../../guides/deployment.md) |
| Policies | [Policies](policies.md) | `eden inference --policy` | [Policy zoo](../../guides/policy-zoo.md), [remote policies](../../guides/remote-policies.md) |
| Retargeting | [Retargeting](retargeting.md) | `eden retarget` | [Retargeting](../../guides/retargeting.md), [dex retargeting](../../guides/dex-retargeting.md) |
| System identification | [System identification](sysid.md) | — | [Planning and sysid](../../guides/planning-and-sysid.md) |
| Motion planning | [Motion planning](motion-planning.md) | — | [Planning and sysid](../../guides/planning-and-sysid.md) |
| Motion optimization | [Motion optimization](motion-optimization.md) | — | [TCRS terrain adaptation](../../guides/tcrs-terrain-adaptation.md) |
| Grasp sampling | [Grasp sampling](grasp-sampling.md) | — | [Grasp sampling](../../guides/grasp-sampling.md) |
| Pose estimation | [Pose estimation](pose-estimation.md) | `eden pose` | [Pose estimation](../../guides/pose-estimation.md) |
| Scripted demonstrations | [Scripted demonstrations](scripted-demos.md) | `eden datagen` | [Scripted demos](../../guides/scripted-demos.md) |
| Teleoperation | [Teleoperation](teleop.md) | `eden teleop` | [Teleoperation](../../guides/teleoperation.md) |
| Visualization | [Visualization](visualization.md) | — | [Sensors and rendering](../../guides/sensors-and-rendering.md) |

## See also

- [CLI reference](../../reference/cli.md)
- [Teleoperation devices](../utils/devices.md)
