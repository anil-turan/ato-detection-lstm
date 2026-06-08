"""
The LSTM model for ATO detection.

An LSTM reads a card's last 10 transactions in order and learns the normal
rhythm of that card. When a new transaction breaks the rhythm, the model gives
a high risk score. Because attacks are very rare, we use class weights so the
model pays much more attention to the few attack examples.

NOTE: TensorFlow is imported lazily inside the functions below. This way the rest
of the project (data, features, sklearn baselines, evaluation) still runs on a
machine where TensorFlow is missing or broken.
"""

import numpy as np

from src import config


def build_model(window_size: int, n_features: int):
    """Create the LSTM network.

    Masking layer  -> ignores the zero-padding we added to short sequences.
    LSTM layers    -> read the sequence and remember patterns over time.
    Dense + sigmoid-> output one risk score between 0 and 1.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        # We track AUC and recall during training because plain accuracy is
        # useless when 99.5% of rows are the same class.
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def class_weights(y: np.ndarray) -> dict:
    """Give the rare attack class a bigger weight, capped so training is stable.

    A pure balanced weight would be huge (attacks are ~0.4%) and makes training
    jumpy. We cap the positive weight at 25, which still strongly favours recall
    without exploding the loss.
    """
    n_pos = max(int(y.sum()), 1)
    n_neg = len(y) - n_pos
    pos_weight = min(n_neg / n_pos, 25.0)
    return {0: 1.0, 1: float(pos_weight)}


def train(X_train, y_train, validation_split=0.15, epochs=25, batch_size=256):
    """Train the LSTM and keep the version with the best validation AUC."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    model = build_model(X_train.shape[1], X_train.shape[2])
    weights = class_weights(y_train)
    print(f"[lstm] class weights: {weights}")

    best_path = config.MODELS_DIR / "lstm_v2_best.keras"
    callbacks = [
        # Stop early if validation AUC stops improving, and restore the best.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_auc", mode="max", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(config.MODELS_DIR / "lstm_v2_final.keras")
    print(f"[lstm] best model saved -> {best_path}")
    return model, history
