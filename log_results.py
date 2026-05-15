"""
Logs the completed training results into MLflow.
Run once after training to populate the UI.
  python log_results.py
"""
import mlflow

TRACKING_URI   = "file:///D:/Personal Projects/Project_2/mlruns"
EXPERIMENT     = "chest-xray-pneumonia"

# Exact metrics from the terminal output
EPOCHS = [
    dict(epoch=1,  train_loss=0.5513, train_auc=0.8511, train_sens=0.8446, train_spec=0.6616, val_loss=0.3871, val_auc=0.9377, val_sens=0.8776, val_spec=0.8198, val_f1=0.9026, lr=1e-4),
    dict(epoch=2,  train_loss=0.3287, train_auc=0.9479, train_sens=0.8681, train_spec=0.8715, val_loss=0.2746, val_auc=0.9606, val_sens=0.8711, val_spec=0.8834, val_f1=0.9100, lr=1e-4),
    dict(epoch=3,  train_loss=0.2402, train_auc=0.9696, train_sens=0.9037, train_spec=0.9102, val_loss=0.1977, val_auc=0.9783, val_sens=0.9171, val_spec=0.9505, val_f1=0.9477, lr=1e-4),
    dict(epoch=4,  train_loss=0.1917, train_auc=0.9787, train_sens=0.9284, train_spec=0.9234, val_loss=0.1798, val_auc=0.9803, val_sens=0.9197, val_spec=0.9081, val_f1=0.9414, lr=1e-4),
    dict(epoch=5,  train_loss=0.1791, train_auc=0.9808, train_sens=0.9384, train_spec=0.9178, val_loss=0.1619, val_auc=0.9851, val_sens=0.9487, val_spec=0.9081, val_f1=0.9569, lr=1e-4),
    dict(epoch=6,  train_loss=0.1482, train_auc=0.9873, train_sens=0.9409, train_spec=0.9395, val_loss=0.1354, val_auc=0.9889, val_sens=0.9276, val_spec=0.9611, val_f1=0.9553, lr=1e-4),
    dict(epoch=7,  train_loss=0.1447, train_auc=0.9873, train_sens=0.9425, train_spec=0.9509, val_loss=0.1321, val_auc=0.9887, val_sens=0.9500, val_spec=0.9576, val_f1=0.9665, lr=1e-4),
    dict(epoch=8,  train_loss=0.1315, train_auc=0.9898, train_sens=0.9464, train_spec=0.9471, val_loss=0.1197, val_auc=0.9911, val_sens=0.9487, val_spec=0.9576, val_f1=0.9658, lr=1e-4),
    dict(epoch=9,  train_loss=0.1335, train_auc=0.9887, train_sens=0.9551, train_spec=0.9357, val_loss=0.1122, val_auc=0.9926, val_sens=0.9500, val_spec=0.9576, val_f1=0.9665, lr=1e-4),
    dict(epoch=10, train_loss=0.1164, train_auc=0.9920, train_sens=0.9544, train_spec=0.9594, val_loss=0.1165, val_auc=0.9918, val_sens=0.9329, val_spec=0.9647, val_f1=0.9588, lr=1e-4),
    dict(epoch=11, train_loss=0.1146, train_auc=0.9918, train_sens=0.9608, train_spec=0.9480, val_loss=0.1048, val_auc=0.9931, val_sens=0.9500, val_spec=0.9611, val_f1=0.9672, lr=1e-4),
    dict(epoch=12, train_loss=0.1072, train_auc=0.9925, train_sens=0.9647, train_spec=0.9584, val_loss=0.0989, val_auc=0.9938, val_sens=0.9645, val_spec=0.9576, val_f1=0.9741, lr=1e-4),
    dict(epoch=13, train_loss=0.1090, train_auc=0.9927, train_sens=0.9599, train_spec=0.9594, val_loss=0.1010, val_auc=0.9937, val_sens=0.9592, val_spec=0.9611, val_f1=0.9720, lr=1e-4),
    dict(epoch=14, train_loss=0.1054, train_auc=0.9930, train_sens=0.9576, train_spec=0.9612, val_loss=0.0980, val_auc=0.9944, val_sens=0.9500, val_spec=0.9717, val_f1=0.9691, lr=1e-4),
    dict(epoch=15, train_loss=0.0973, train_auc=0.9940, train_sens=0.9624, train_spec=0.9679, val_loss=0.0977, val_auc=0.9939, val_sens=0.9605, val_spec=0.9682, val_f1=0.9740, lr=1e-4),
    dict(epoch=16, train_loss=0.0995, train_auc=0.9937, train_sens=0.9624, train_spec=0.9594, val_loss=0.1199, val_auc=0.9918, val_sens=0.9711, val_spec=0.9470, val_f1=0.9755, lr=1e-4),
    dict(epoch=17, train_loss=0.0897, train_auc=0.9949, train_sens=0.9644, train_spec=0.9679, val_loss=0.0989, val_auc=0.9948, val_sens=0.9395, val_spec=0.9788, val_f1=0.9649, lr=1e-4),
    dict(epoch=18, train_loss=0.0868, train_auc=0.9953, train_sens=0.9663, train_spec=0.9622, val_loss=0.0906, val_auc=0.9951, val_sens=0.9500, val_spec=0.9717, val_f1=0.9691, lr=1e-4),
    dict(epoch=19, train_loss=0.0791, train_auc=0.9960, train_sens=0.9714, train_spec=0.9716, val_loss=0.0898, val_auc=0.9951, val_sens=0.9645, val_spec=0.9647, val_f1=0.9754, lr=1e-4),
    dict(epoch=20, train_loss=0.0795, train_auc=0.9960, train_sens=0.9721, train_spec=0.9669, val_loss=0.0901, val_auc=0.9950, val_sens=0.9592, val_spec=0.9647, val_f1=0.9726, lr=1e-4),
]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="resnet18_baseline"):
    mlflow.log_params({
        "model":            "resnet18",
        "epochs":           20,
        "batch_size":       32,
        "learning_rate":    1e-4,
        "weight_decay":     1e-4,
        "optimizer":        "AdamW",
        "freeze_layers":    6,
        "dropout":          0.3,
        "mixed_precision":  True,
        "image_size":       224,
        "dataset":          "chest_xray_kaggle",
        "train_samples":    4172,
        "val_samples":      1044,
        "test_samples":     624,
    })

    for row in EPOCHS:
        e = row["epoch"]
        mlflow.log_metrics({
            "train_loss":        row["train_loss"],
            "train_auc":         row["train_auc"],
            "train_sensitivity": row["train_sens"],
            "train_specificity": row["train_spec"],
            "val_loss":          row["val_loss"],
            "val_auc":           row["val_auc"],
            "val_sensitivity":   row["val_sens"],
            "val_specificity":   row["val_spec"],
            "val_f1":            row["val_f1"],
            "learning_rate":     row["lr"],
        }, step=e)
        print(f"  Epoch {e:>2} logged.")

    # Test set final results
    mlflow.log_metrics({
        "test_auc":         0.9620,
        "test_accuracy":    0.8622,
        "test_sensitivity": 0.9949,
        "test_specificity": 0.6410,
        "test_f1":          0.9002,
        "test_tp":          388,
        "test_fp":          84,
        "test_tn":          150,
        "test_fn":          2,
    }, step=20)

    mlflow.set_tag("best_epoch",    "18")
    mlflow.set_tag("best_val_auc",  "0.9951")
    mlflow.set_tag("status",        "complete")

print("\nDone. All 20 epochs + test results logged to MLflow.")
print(f"Run: {TRACKING_URI}")
