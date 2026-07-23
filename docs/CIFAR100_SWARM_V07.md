# CIFAR-100 Swarm Campaign — v0.7.0

## Objective

The campaign's objective is not merely to distribute generic training. It is to exploit the Dendritron's owned, sparse tissues so ordinary machines can search many bounded improvements while the canonical model remains auditable.

The two explicit optimization targets are:

1. **model quality** — train the sensory field, nonlinear experts, gates, readouts, repair paths, and associative memory; and
2. **routing quality** — reduce the difference between complete-model accuracy and correct-category oracle accuracy.

## Native class ownership

CIFAR-100 provides 100 fine labels and 20 coarse labels. The preparation layer derives an explicit fine-to-coarse relation from the archive. Native class IDs are reordered so each Native10 colony owns exactly the five fine classes belonging to one official coarse category.

The mapping is stored in the dataset manifest:

- `official_to_native`
- `native_to_official`
- `fine_labels_by_coarse_category`
- fine and coarse label names

No assumption is made that official fine IDs are already contiguous by category.

## Sensory adapter

The official Python rows are channel-major: 1,024 red values, then green, then blue. v0.7 normalizes each channel and reorders the image into a 4×2 grid of spatial patches. Each patch contains 8×16 pixels across RGB, or 384 values. The eight consecutive 384-value blocks are therefore spatially meaningful inputs to the eight sensory field tissues.

For field-training tasks, the coordinator sends augmented uint8 rows plus the committed normalization metadata. The worker deterministically derives the float32 patch tensor locally. This avoids expanding a 640-image field shard to roughly 7.5 MiB of raw float32 before JSON/base64 overhead.

## Training data and evidence data

The default split is stratified per fine class:

- 450 training images;
- 25 selection-bank images;
- 25 replication-bank images;
- all 100 official test images remain test-only.

The default five images per class per round permits five statistically guarded campaign rounds. Operators may select a different committed fold size before the campaign begins. Smaller folds permit more rounds but reduce statistical power. The implementation reports this tradeoff and never silently reuses a consumed fold.

A longer campaign requires a new, trainer-invisible validation bank from a documented source. Repartitioning already exposed training images into a supposedly fresh holdout is not treated as fresh evidence.

## Routing-gap report

For a balanced training diagnostic sample, the coordinator measures:

- complete routed-model accuracy;
- correct-category oracle accuracy;
- oracle routing gap;
- category recall at top 1, 2, 4, 8, and 20;
- recall after bounded low-margin expansion;
- route misses and accuracy on route misses;
- accuracy conditional on the correct category being routed;
- correct-category rank distribution;
- per-category versions of the same quantities.

The planner is prohibited from accepting a report whose split is `test`.

## Routing search

When category routing is limiting, the planner searches:

- hard-negative routing margins;
- scout diversity strengths;
- learning-rate and step-budget combinations;
- field-level routing margins and margin weights;
- the category with the largest route-miss burden;
- periodic sensory-field interventions for global routing geometry.

Each candidate has a committed `search_recipe` and `search_recipe_hash`. A mismatch is rejected by the result schema.

## Colony training

When routing is sufficiently accurate, the planner cycles through:

- `expert_train`
- `branch_train`
- `memory_train`

and targets the category with the weakest correct-category oracle performance. Expert and branch operations retain the rotating 15-of-45 ownership schedule.

## Promotion and final test

Selection and replication are distinct all-class artifacts. Candidate families are precommitted. The coordinator recomputes exact paired evidence and applies the McNemar, effect-size, and class-harm gates. Replay and replication use identities excluded from candidate generation.

The official test split is evaluated only after a snapshot is frozen. Test results do not feed the planner, rewards, candidate ranking, or model composition.

## Distributed-compute evidence

Each promoted round records:

- candidate-search count;
- trainer and verifier report counts;
- contributed worker seconds and hours;
- selected candidate and exact sparse delta;
- search yield;
- routing metrics before and after promotion;
- selection and replication evidence.

These measurements allow the live campaign to answer the actual economic question: how much benchmark gain is produced per donated worker-hour, verifier-hour, and transferred byte.

## Current claim boundary

v0.7 supplies the real CIFAR-100 campaign implementation. The packaging environment did not contain the official archive, so the release does not claim a new CIFAR-100 accuracy number. The first public campaign report must include the prepared dataset manifest, imported or initialized model root, every promoted contribution, the untouched official-test report, and the external baseline evidence hash.
