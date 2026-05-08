# Requirements v1.0 (locked 2026-04-14)

## 1. Project Overview

| Item | Value |
|---|---|
| Purpose | Privacy protection and expressive freedom for live streaming |
| User | Single streamer (Momo) - solo use only |
| Dev host | Windows 10 Pro, RTX 5080, 32 GB RAM |
| Streaming host | Separate PC (to be prepared) |
| Camera | OBSBOT Meet 2 4K (built-in filters disabled) |
| Streaming software | OBS Studio consuming our virtual camera |

## 2. Must-Have Requirements

### 2.1 Safety (highest priority)

- **R-S1**: While a face is detected in the camera frame, the face transformation MUST NEVER drop. A single frame of the unmasked real face is a critical failure.
- **R-S2**: When the face leaves frame, detection confidence drops below threshold, or any failsafe condition fires, the output MUST be a solid **black frame**.
- **R-S3**: When hands or objects partially occlude the face, the swapped face MUST NOT be rendered onto the occluder ("ghost face" prohibition).

### 2.2 Occlusion behavior

- **R-O1**: When a hand partially covers the face, the user's real hand SHALL be displayed as-is, and the swap SHALL continue on the still-visible face region.
- **R-O2**: Partial occlusion by hands/objects MUST NEVER trigger the freeze/black-frame failsafe. Freeze is reserved exclusively for total face-loss / low-confidence conditions.

### 2.3 Quality

- **R-Q1**: Mouth and facial movement remain tightly synchronized - no perceptible lip-sync drift.
- **R-Q2**: Natural appearance at close range (face filling a large portion of frame).
- **R-Q3**: "AI-looking" output is acceptable; natural is preferred.

### 2.4 OBS integration

- **R-I1**: Output to a virtual camera consumable by OBS (`pyvirtualcam` / OBS Virtual Cam plugin).

### 2.5 Performance

- **R-P1**: Capture resolution 1920 x 1080.
- **R-P2**: Sustained 30 FPS output.
- **R-P3**: Internal pipeline latency <= 100 ms (capture -> virtual camera write).

### 2.6 Identity source

- **R-D1**: Identity derived from the 6 reference images in `source_faces/` (AI-generated face).
- **R-D2**: A trained `.dfm` file will be produced (Phase 3) via DeepFaceLab SAEHD from dataset augmented from the 6 stills plus Momo's recorded face video.

## 3. Nice-to-Have Requirements

- **R-N1**: Coexistence with an external voice changer (no direct integration, just non-interference).
- **R-N2**: Skin smoothing applied ONLY to the user's own visible skin (forehead margin / jaw / neck around the swap region). The swapped face itself is NOT touched by skin smoothing.

## 4. Technical Constraints

- **C-1**: ~10 s broadcast delay between OBS and viewer is acceptable (independent of our internal latency).
- **C-2**: Lip/face desync is unacceptable.
- **C-3**: Solo user, single PC deployment. No cloud, no multi-user, no networked services.

## 5. Resolved Design Decisions

| Question | Decision |
|---|---|
| Failsafe action on trigger | **Solid black frame** |
| Freeze on partial occlusion | **Never** - partial occlusion handled by masking, freeze is for face-loss only |
| Skin filter scope | **User's own visible skin only**, not the swapped face |
| Internal latency budget | **<= 100 ms** |
| Face-swap technology | **Trained `.dfm` via DeepFaceLive runtime** (user-mandated; no one-shot swapper alternatives) |
| Source images | The 6 PNGs in `source_faces/` will drive Phase 2a data augmentation |
