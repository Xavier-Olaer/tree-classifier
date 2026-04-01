# tf_train_improved.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =======================
# SETTINGS
# =======================
BATCH_SIZE = 16   # smaller batch = better for small dataset
EPOCHS = 15
IMG_SIZE = (224, 224)

train_dir = "fruit_tree_dataset/train"
val_dir = "fruit_tree_dataset/validation"

# =======================
# DATA AUGMENTATION (STRONGER)
# =======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# =======================
# MODEL
# =======================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # helps prevent overfitting
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# =======================
# STEP 1: Freeze most layers
# =======================
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Training (transfer learning phase)...")

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# =======================
# STEP 2: Fine-tune ALL layers
# =======================
print("🔥 Fine-tuning entire model...")

base_model.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# =======================
# SAVE MODEL
# =======================
model.save("tree_classifier_tf")
print("✅ TensorFlow model saved!")

# =======================
# CONVERT TO TFLITE
# =======================
converter = tf.lite.TFLiteConverter.from_saved_model("tree_classifier_tf")
tflite_model = converter.convert()

with open("tree_classifier_v2.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as tree_classifier_v2.tflite")