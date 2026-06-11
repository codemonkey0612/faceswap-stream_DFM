# Akool Live Face Swap → OBS → Stripchat

Browser-based live face swap using Akool's Live Face Swap API + Agora.
The page publishes your webcam to Akool, receives the swapped video, and shows
it fullscreen so OBS can capture it into the Virtual Camera.

## Why a browser page (not Python)

Akool's live swap runs on Agora (WebRTC), which is browser-native. Running it
in the browser and capturing with OBS is far more reliable than reimplementing
WebRTC in Python.

## One-time setup

1. **Install OBS** (already done) and make sure **Virtual Camera** works.
2. Put your **source face image** on a public URL (Google Drive direct-download
   form works: `https://drive.google.com/uc?export=download&id=FILE_ID`).
3. **Close OBSBOT Center** so the webcam isn't locked (same as the DFM pipeline).

## Run it

1. Open `akool/live_faceswap.html` in **Google Chrome**
   (double-click the file, or drag it into Chrome).
2. Fill in:
   - **Client ID** and **API Key** (from the Akool dashboard)
   - **Source face image URL** (the public image of the character)
   - **Camera device** (optional — type part of the webcam name, e.g. `FHD`)
3. Click **Start**. Allow camera access.
4. When status shows **"LIVE ✓ … swapped stream received"**, the swapped face
   fills the window.
5. Click **Hide panel** so only the clean swapped video is visible.

## Send it to Stripchat via OBS

1. In OBS, add a **Window Capture** (or **Browser** if you host the file) source
   pointing at the Chrome window running this page.
2. Resize/crop so the swapped video fills the OBS canvas.
3. OBS → **Start Virtual Camera**.
4. In Stripchat broadcast settings, pick **OBS Virtual Camera**.
5. Audio: route the RVC/voice-changer output as the microphone (separate from
   this page — Akool audio is ignored here).

## Cost / billing safety

- Akool bills per live second (~10 credits / 30s).
- The page **closes the session automatically** on **Stop** and on closing the
  tab/window, so billing stops. Always click **Stop** when done.
- Free trial gives ~30s per session for testing.

## Notes / limits

- Live resolution is **640x480 @ 20fps** (Akool's recommended live config).
- There is cloud round-trip latency (~150-300ms) — keep audio in sync.
- Source image / face landmarks are detected automatically via Akool's
  detect API when you click Start.

## Quick API test (no browser)

`scripts/akool_session.sh` (curl) creates and closes a session for debugging
the credentials / source URL without the UI. See that file.
