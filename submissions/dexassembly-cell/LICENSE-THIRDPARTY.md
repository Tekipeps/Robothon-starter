# Third-party assets

## LEAP Hand (`assets/leap_hand/`)

The 16-DOF LEAP Hand MJCF model and meshes are vendored from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/leap_hand)
by Google DeepMind, which packages the model released by the LEAP Hand authors
(Shaw, Agarwal, Pathak — *LEAP Hand: Low-Cost, Efficient, and Anthropomorphic
Hand for Robot Learning*, RSS 2023).

The model is distributed under the **MIT License**; the original license text is
retained at [`assets/leap_hand/LICENSE`](assets/leap_hand/LICENSE).

The hand is committed into this submission verbatim so the project is fully
reproducible without any runtime download. All other scene elements (the gantry,
bench, bins, parts, peg-and-hole fixture, and inspection button) are generated
procedurally by `dexassembly/scene.py` and are original to this submission.
