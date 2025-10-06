# neural_network_visualizer.py (Refactored)

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# --- CONFIGURATION ---
MODEL_PATH = "mnist_model.keras"
MAX_NODES_TO_VISUALIZE = 16  # Max nodes to show per layer in the visualization

# --- DATA HANDLING ---
def load_mnist_data():
    """Loads and preprocesses the MNIST dataset for a CNN."""
    mnist = keras.datasets.mnist
    (train_images, train_labels), _ = mnist.load_data()
    # Normalize and expand dimensions for CNN
    train_images = train_images.astype("float32") / 255.0
    train_images = np.expand_dims(train_images, -1)
    return train_images, train_labels

# --- MODEL MANAGEMENT ---
def create_cnn_model():
    """Defines the Convolutional Neural Network (CNN) architecture."""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model():
    """Trains the CNN model with data augmentation and saves it."""
    with st.spinner("Training a new CNN model with data augmentation. This is key for robustness..."):
        train_images, train_labels = load_mnist_data()
        model = create_cnn_model()

        # --- Data Augmentation ---
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        datagen.fit(train_images)

        # Train the model using the augmented data generator
        model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                  epochs=10,  # Increased epochs for augmented data
                  verbose=0)
        
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        model.save(MODEL_PATH)
    st.success(f"CNN model trained with augmentation and saved to '{MODEL_PATH}'")
    return model

@st.cache_resource
def load_model():
    """
    Loads the saved Keras model from disk. If the model is not found,
    it triggers the training process.
    """
    if os.path.exists(MODEL_PATH):
        st.info(f"Loading cached CNN model from '{MODEL_PATH}'...")
        return keras.models.load_model(MODEL_PATH)
    else:
        st.warning("No pre-trained model found. Training a new one.")
        return train_and_save_model()

@st.cache_resource
def get_activation_model(_model):
    """
    Creates a new model from an existing one to output intermediate activations.
    Uses the Keras Functional API to build the activation model.
    """
    # The input shape must match the model's expected input
    inputs = keras.Input(shape=(28, 28, 1))
    x = inputs
    layer_outputs = []
    for layer in _model.layers:
        x = layer(x)
        layer_outputs.append(x)
    return keras.models.Model(inputs=inputs, outputs=layer_outputs)

# --- VISUALIZATION ---
def display_feature_maps(activations, model_layers):
    """
    Displays the feature maps from convolutional and pooling layers.

    Args:
        activations (list): List of all layer activations.
        model_layers (list): List of all model layers.
    """
    st.write("### Feature Map Activations")
    st.info("Feature maps from convolutional layers show learned patterns like edges and textures.")

    for layer, activation in zip(model_layers, activations):
        if not isinstance(layer, (keras.layers.Conv2D, keras.layers.MaxPooling2D)):
            continue

        st.write(f"**Layer: `{layer.name}`** (Output Shape: {activation.shape})")

        activation_maps = activation[0]
        num_maps = activation_maps.shape[-1]
        maps_to_show = min(num_maps, 16)

        cols = st.columns(8)
        for i in range(maps_to_show):
            with cols[i % 8]:
                feature_map = activation_maps[:, :, i]
                # Robust normalization for better visualization
                p2, p98 = np.percentile(feature_map, (2, 98))
                feature_map = np.clip((feature_map - p2) / (p98 - p2 + 1e-6), 0, 1)
                st.image(feature_map, caption=f'Map {i+1}', use_column_width=True)
        st.markdown("---")

def draw_neural_network(ax, activations):
    """
    Draws the dense part of the neural network and highlights neuron activations.
    Subsamples nodes for large layers to maintain visual clarity.
    """
    ax.clear()
    ax.axis('off')
    ax.set_title("Dense Layer Activations", fontsize=14)

    layer_sizes = [arr.shape[-1] for arr in activations]
    num_layers = len(layer_sizes)
    layer_xs = np.linspace(0, 1, num_layers)
    cmap = plt.get_cmap('viridis')

    node_ys_viz, node_indices_viz = [], []
    for size in layer_sizes:
        if size > MAX_NODES_TO_VISUALIZE:
            indices = np.linspace(0, size - 1, MAX_NODES_TO_VISUALIZE).astype(int)
            node_indices_viz.append(indices)
            node_ys_viz.append(np.linspace(0, 1, MAX_NODES_TO_VISUALIZE))
        else:
            node_indices_viz.append(np.arange(size))
            node_ys_viz.append(np.linspace(0, 1, size))
            
    for i in range(num_layers - 1):
        for j_idx in range(len(node_ys_viz[i])):
            for k_idx in range(len(node_ys_viz[i+1])):
                ax.plot([layer_xs[i], layer_xs[i+1]], [node_ys_viz[i][j_idx], node_ys_viz[i+1][k_idx]], 'k-', alpha=0.05, lw=0.5)

    for i, (activation_layer, size) in enumerate(zip(activations, layer_sizes)):
        activation_values = activation_layer.flatten()
        current_indices = node_indices_viz[i]
        current_ys = node_ys_viz[i]

        if len(current_indices) < size:
            mid_y_idx = len(current_ys) // 2
            mid_y = (current_ys[mid_y_idx-1] + current_ys[mid_y_idx]) / 2
            ax.text(layer_xs[i], mid_y, '⋮', ha='center', va='center', fontsize=16, color='gray')

        for j_viz_idx, j_original_idx in enumerate(current_indices):
            activation = activation_values[j_original_idx]
            color = cmap(activation)
            radius = 0.02 if size <= MAX_NODES_TO_VISUALIZE else 0.015
            circle = plt.Circle((layer_xs[i], current_ys[j_viz_idx]), radius, color=color, ec='k', zorder=4)
            ax.add_patch(circle)
            
            if i == num_layers - 1:
                ax.text(layer_xs[i] + 0.08, current_ys[j_viz_idx], f"{j_original_idx}", ha='center', va='center', fontsize=10)

# --- STREAMLIT UI ---
def setup_ui():
    """Sets up the Streamlit page layout and interactive elements."""
    st.set_page_config(layout="wide")
    st.title("🧠 Interactive CNN Visualization for MNIST")
    st.markdown("Draw a digit to see the CNN's feature maps and predictions!")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Input Canvas")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)", stroke_width=20, stroke_color="#FFFFFF",
            background_color="#000000", height=280, width=280,
            drawing_mode="freedraw", key="canvas",
        )

    with col2:
        st.header("Visualization & Output")
        prediction_placeholder = st.empty()
        feature_map_placeholder = st.empty()
        plot_placeholder = st.empty()

    return canvas_result, plot_placeholder, prediction_placeholder, feature_map_placeholder

def preprocess_canvas_image(canvas_result):
    """Converts canvas drawing to a model-compatible numpy array for the CNN."""
    img_data = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img_data, 'RGBA').convert('L')
    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized) / 255.0
    return img_array.reshape(1, 28, 28, 1)

def main():
    """Main function to run the Streamlit application."""
    canvas_result, plot_placeholder, prediction_placeholder, feature_map_placeholder = setup_ui()
    
    model = load_model()
    activation_model = get_activation_model(model)

    flatten_idx = next(i for i, layer in enumerate(model.layers) if isinstance(layer, keras.layers.Flatten))
    dense_layer_outputs = activation_model.outputs[flatten_idx:]

    fig, ax = plt.subplots(figsize=(8, 6))
    initial_dense_activations = [np.zeros((1, output.shape[-1])) for output in dense_layer_outputs]
    draw_neural_network(ax, initial_dense_activations)
    plot_placeholder.pyplot(fig)

    if st.button("✨ Predict") and canvas_result.image_data is not None:
        img_reshaped = preprocess_canvas_image(canvas_result)
        activations = activation_model.predict(img_reshaped)

        dense_activations = activations[flatten_idx:]
        draw_neural_network(ax, dense_activations)
        plot_placeholder.pyplot(fig)
        plt.close(fig)

        with feature_map_placeholder.container():
            display_feature_maps(activations, model.layers)
        
        with prediction_placeholder.container():
            final_predictions = activations[-1][0]
            predicted_digit = np.argmax(final_predictions)
            confidence = np.max(final_predictions)

            st.metric(label="Predicted Digit", value=str(predicted_digit), delta=f"{confidence:.2%} confidence")
            st.write("**Confidence Scores per Digit:**")
            st.bar_chart(final_predictions)

if __name__ == "__main__":
    main()