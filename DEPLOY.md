# Putting the Deepfake Detector live

The site is plain files: no server. The model (`celeb_model.pth`, converted to
`celeb_model.onnx`) runs inside each visitor's browser, GitHub Pages hosts it
for free, and a free Firebase database holds the shared confusion matrix.

The live link will be:

    https://jamesserrano94.github.io/DeepfakeDetector/

## What's in this folder

| Path | What it is |
|---|---|
| `docs/` | The website. GitHub Pages serves this folder. |
| `docs/config.js` | The only file you edit: Firebase keys, your name, validation numbers. |
| `docs/model/celeb_model.onnx` | Your SimpleCNN, converted from `celeb_model.pth`. |
| `docs/lib/` | ONNX Runtime Web and the Firebase SDK, bundled so nothing loads from a CDN. |
| `firestore.rules` | Security rules for the shared matrix. Paste into Firebase. |
| `export_onnx.py` | Rebuilds the `.onnx` file if you retrain. |

## Part 1: Get it live (about 5 minutes)

1. Unzip the download.
2. Open https://github.com/JamesSerrano94/DeepfakeDetector, then
   **Add file > Upload files**.
3. Drag in the whole `docs` folder, plus `export_onnx.py`, `firestore.rules`
   and this `DEPLOY.md`. Dragging the folder keeps its structure. Click
   **Commit changes**.
4. Go to **Settings > Pages**. Under *Build and deployment*, set Source to
   **Deploy from a branch**, Branch to **main**, folder to **/docs**, then **Save**.
5. Wait a minute or two and refresh. The page shows "Your site is live at ...".

It works now. Until Part 2 is done, the confusion matrix only counts on each
visitor's own device, and the line above it says so.

## Part 2: Share the matrix across visitors (about 10 minutes, free)

1. Go to https://console.firebase.google.com and sign in with your Google account.
2. **Create a project**. Any name works (for example `deepfake-detector`).
   You can turn off Google Analytics and Gemini; neither is needed.
3. Open **Firestore Database** from the left menu and click **Create database**.
   - If asked for an edition, pick **Standard**.
   - Leave the database ID as **(default)**. The site looks for that name.
   - Pick a US location. It can't be changed later, but any US location is fine.
   - Choose **Start in production mode**.
4. On the database page, open the **Rules** tab. Replace everything there
   with the contents of `firestore.rules` and click **Publish**.
5. Click the gear next to *Project Overview*, then **Project settings**. Under
   *Your apps*, click the web icon `</>`. Give it a nickname, leave
   *Firebase Hosting* unticked, and click **Register app**.
6. Copy the `firebaseConfig = { ... }` object it shows you.
7. In your GitHub repo, open `docs/config.js`, click the pencil icon, and
   replace `firebase: null,` with the object you copied, for example:

   ```js
   firebase: {
     apiKey: "AIza...",
     authDomain: "deepfake-detector-xxxx.firebaseapp.com",
     projectId: "deepfake-detector-xxxx",
     storageBucket: "deepfake-detector-xxxx.firebasestorage.app",
     messagingSenderId: "1234567890",
     appId: "1:1234567890:web:abc123"
   },
   ```

   Keep the comma after the closing brace. Commit.
8. After a minute, reload the site. The line above the matrix should say
   "shared by everyone who visits". Drop in one video with a label; the count
   goes up, and a document appears under **Firestore Database > Data**.

About the API key: Firebase web keys are meant to be public, and the security
rules are what protect the database. If GitHub emails you about an "exposed
Google API key", that's why; you can close the alert.

## Checking it worked

| You see | It means |
|---|---|
| "Model ready" at the top right | The model loaded in the browser. |
| "shared by everyone who visits" above the matrix | Firebase is connected. |
| "counted on this device only" | `firebase` in `config.js` is still `null`. |
| "shared matrix unavailable" | Firebase is set up but refusing. Check the rules were published and the database ID is `(default)`. Open the browser console (F12) for the exact error. |

Every analysis is also saved to the `predictions` collection (probability,
verdict, the visitor's label, frame count, time), which you can browse or
export from the Firebase console. The video itself is never uploaded.

## If you retrain the model

1. Put the new `celeb_model.pth` in the repo root.
2. Run `python export_onnx.py` (needs `pip install torch onnx onnxruntime`).
   It rewrites `docs/model/celeb_model.onnx` and checks it matches PyTorch.
3. Commit the new `.onnx` file.

`export_onnx.py` and `docs/preprocess.js` assume the SimpleCNN setup from the
notebook: 112x112 RGB frames, `Resize` then `ToTensor`, no normalization, one
output logit where 1 means fake. If you switch to one of the transfer-learning
models, the architecture in `export_onnx.py` and the input size in
`docs/preprocess.js` both need to change to match.

## Adding it to your README and portfolio

Add a line near the top of the repo README:

```markdown
**Live demo:** https://jamesserrano94.github.io/DeepfakeDetector/
```

On your portfolio, link the project card to the same URL, with the repo as a
second link.
