// Site settings. This is the only file you should need to edit.
window.SITE_CONFIG = {
  author: "James Serrano",
  repoUrl: "https://github.com/JamesSerrano94/DeepfakeDetector",

  // Paste the firebaseConfig object from your Firebase project here
  // (Project settings > General > Your apps > Web app). Until you do, the
  // page still works, but the matrix only counts on each visitor's own device.
  //
  // It should end up looking like this (your values will differ):
  //   firebase: {
  //     apiKey: "AIza...",
  //     authDomain: "your-project.firebaseapp.com",
  //     projectId: "your-project",
  //     storageBucket: "your-project.firebasestorage.app",
  //     messagingSenderId: "1234567890",
  //     appId: "1:1234567890:web:abc123"
  //   },
  firebase: null,

  // Your saved validation results (the confusion matrix in Untitled.png).
  // Set this to null to hide the line under the matrix.
  validation: {
    tn: 912, fp: 5, fn: 6, tp: 794,
    auc: 1.0,
    caveat:
      "That split was made frame by frame, so frames from the same video " +
      "appear in both training and validation. Expect lower accuracy on " +
      "videos the model has never seen."
  }
};
