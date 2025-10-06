# neural_network_visualizer.py (Corrected for Keras 3.x Model Loading)

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# --- MODEL TRAINING & LOADING ---
MODEL_PATH = "mnist_model.keras"
MAX_NODES_TO_VISUALIZE = 16

def create_model():
    """Defines the neural network architecture."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model():
    """Trains the model on the MNIST dataset and saves it."""
    with st.spinner("Training a new model on the MNIST dataset. This will run only once..."):
        mnist = keras.datasets.mnist
        (train_images, train_labels), _ = mnist.load_data()
        train_images = train_images / 255.0

        model = create_model()
        model.fit(train_images, train_labels, epochs=5, verbose=0)
        
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        model.save(MODEL_PATH)
    st.success(f"Model trained and saved to '{MODEL_PATH}'")
    return model

@st.cache_resource
def load_model():
    """Loads the saved model from disk, or trains a new one."""
    if os.path.exists(MODEL_PATH):
        print("Loading existing model.")
        return keras.models.load_model(MODEL_PATH)
    else:
        print("Model not found. Training a new one.")
        return train_and_save_model()

# --- VISUALIZATION FUNCTION ---

def draw_neural_network(ax, activations):
    """Draws the neural network with activations highlighted."""
    ax.clear()
    ax.axis('off')

    layer_sizes = [arr.shape[-1] for arr in activations]
    num_layers = len(layer_sizes)
    
    layer_xs = np.linspace(0, 1, num_layers)
    cmap = plt.get_cmap('viridis')

    node_ys_viz = []
    node_indices_viz = []
    for size in layer_sizes:
        if size > MAX_NODES_TO_VISUALIZE:
            indices = np.concatenate([
                np.arange(0, MAX_NODES_TO_VISUALIZE // 2),
                np.arange(size - MAX_NODES_TO_VISUALIZE // 2, size)
            ]).astype(int)
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
    
    ax.set_title("Neural Network Activations", fontsize=14)

# --- STREAMLIT APP ---

st.set_page_config(layout="wide")
st.title("🧠 Interactive Neural Network Visualization")
st.markdown("Draw a digit on the canvas and click **Predict** to see the neurons activate!")

# Load the model
model = load_model()

# --- Corrected Activation Model Creation ---
# Rebuild the model using the Keras Functional API to ensure the input is defined.
# 1. Define an explicit input tensor
inputs = keras.Input(shape=(28, 28))

# 2. Reuse the layers from the loaded model to build a new graph
x = inputs
layer_outputs = []
for layer in model.layers:
    x = layer(x)
    layer_outputs.append(x)

# 3. Create the new model that outputs the activations of every layer
activation_model = keras.models.Model(inputs=inputs, outputs=layer_outputs)

# App layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    predict_button = st.button("✨ Predict")

with col2:
    st.header("Visualization & Output")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_placeholder = st.empty()
    prediction_placeholder = st.empty()
    
    initial_activations = [np.zeros((1, output.shape[-1])) for output in activation_model.outputs]
    draw_neural_network(ax, initial_activations)
    plot_placeholder.pyplot(fig)
    plt.close(fig) # Close figure right after initial plot to save memory

if predict_button and canvas_result.image_data is not None:
    # Preprocess the image
    img_data = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img_data, 'RGBA').convert('L')
    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img_resized) / 255.0
    img_reshaped = img_array.reshape(1, 28, 28)

    # Get activations
    activations = activation_model.predict(img_reshaped)
    
    # Redraw the plot with new activations
    fig, ax = plt.subplots(figsize=(8, 6)) # Create a new figure for the update
    draw_neural_network(ax, activations)
    plot_placeholder.pyplot(fig)
    plt.close(fig)
    
    # Display Prediction
    with prediction_placeholder.container():
        final_predictions = activations[-1][0]
        predicted_digit = np.argmax(final_predictions)
        confidence = np.max(final_predictions)
        
        st.metric(label="Predicted Digit", value=f"{predicted_digit}", delta=f"{confidence:.2%} confidence")
        st.write("**Confidence Scores per Digit:**")
        st.bar_chart(final_predictions)