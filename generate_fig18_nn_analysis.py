import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_neural_network_digital_twin():
    # ==========================================
    # 1. CONFIGURATION FROM MATLAB PIPELINE.M
    # ==========================================
    
    # Input Layer: Top 15 Features (Representative selection from MRMR step)
    input_labels = [
        'RMS', 'Kurtosis', 'Skewness', 
        'Crest Factor', 'Spectral Centroid', 
        'BandEnergy (OilWhirl)', 'BandEnergy (Misalign)',
        'BandEnergy (Cavitation)', 'Harmonic 2X Ratio',
        'Harmonic 3X Ratio', 'Spectral Entropy',
        'Wavelet Kurtosis', 'Bispectrum Peak',
        'Hilbert StdDev', 'Cepstral Peak'
    ]
    
    # Output Layer: Exact 11 Classes from pipeline.m header
    output_labels = [
        'Healthy (Sain)', 
        'Misalignment', 
        'Imbalance', 
        'Looseness (Jeu)', 
        'Lubrication', 
        'Cavitation', 
        'Wear (Usure)', 
        'Oil Whirl', 
        'Mixed: Misalign+Imbal', 
        'Mixed: Wear+Lube', 
        'Mixed: Cavit+Jeu'
    ]
    
    # Architecture defined in Step 4.8 of pipeline.m
    # LayerSizes = [50, 25], Activations = 'relu'
    layer_sizes = [15, 50, 25, 11] 
    layer_names = ['Input Layer\n(15 Selected Features)', 
                   'Hidden Layer 1\n(50 Neurons - ReLU)', 
                   'Hidden Layer 2\n(25 Neurons - ReLU)', 
                   'Output Layer\n(11 Fault Classes)']
    
    # ==========================================
    # 2. DRAWING LOGIC
    # ==========================================
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Layout parameters
    left, right = 0.1, 0.9
    bottom, top = 0.1, 0.9
    h_spacing = (right - left) / (len(layer_sizes) - 1)
    
    # Colors
    colors = ['#3498db', '#e74c3c', '#e74c3c', '#2ecc71'] # Blue, Red, Red, Green
    
    # Store node coordinates for drawing edges
    node_coords = []
    
    for l_idx, size in enumerate(layer_sizes):
        layer_x = left + l_idx * h_spacing
        layer_nodes = []
        
        # Determine how many nodes to draw (truncate if too large)
        # We draw the top N and bottom N, and put "..." in middle
        max_visual_nodes = 16
        if size > max_visual_nodes:
            top_nodes = 8
            bottom_nodes = 8
            indices = list(range(top_nodes)) + list(range(size - bottom_nodes, size))
            has_break = True
        else:
            indices = range(size)
            has_break = False
            
        # Vertical spacing calculation
        # Use a fixed height for the visual representation to keep layers aligned
        visual_height = 0.8
        v_spacing = visual_height / (len(indices) + (1 if has_break else -1))
        center_y = (top + bottom) / 2
        
        # Draw Nodes
        current_y = center_y + (len(indices) * v_spacing) / 2 - v_spacing/2
        
        for i, real_idx in enumerate(indices):
            # If we are at the break point (middle of split)
            if has_break and i == top_nodes:
                ax.text(layer_x, current_y, "⋮", fontsize=25, ha='center', va='center', fontweight='bold', color='gray')
                current_y -= v_spacing
            
            # Draw Circle
            circle = patches.Circle((layer_x, current_y), radius=0.018, 
                                    facecolor='white', edgecolor=colors[l_idx], 
                                    linewidth=2, zorder=4)
            ax.add_patch(circle)
            layer_nodes.append((layer_x, current_y))
            
            # Add Labels
            # Input Layer Labels (Left)
            if l_idx == 0:
                label_text = input_labels[real_idx]
                ax.text(layer_x - 0.04, current_y, label_text, ha='right', va='center', fontsize=10, fontweight='bold')
            
            # Output Layer Labels (Right)
            elif l_idx == len(layer_sizes) - 1:
                label_text = output_labels[real_idx]
                ax.text(layer_x + 0.04, current_y, label_text, ha='left', va='center', fontsize=10, fontweight='bold')
                
            current_y -= v_spacing

        node_coords.append(layer_nodes)
        
        # Add Layer Title
        ax.text(layer_x, top + 0.05, layer_names[l_idx], ha='center', va='bottom', fontsize=12, fontweight='bold')

    # ==========================================
    # 3. DRAW CONNECTIONS (WEIGHTS)
    # ==========================================
    for l in range(len(node_coords) - 1):
        layer_a = node_coords[l]
        layer_b = node_coords[l+1]
        
        # Draw connections between visible nodes
        for i, (xa, ya) in enumerate(layer_a):
            for j, (xb, yb) in enumerate(layer_b):
                # Logic to make lines cleaner:
                # Don't draw every single line if it's the "break" layer to avoid clutter
                # only draw a subset to imply full connectivity
                
                alpha = 0.7 # Transparency
                if l == 1 or l == 2: # Hidden layers
                     alpha = 0.7
                
                line = patches.ConnectionPatch(xyA=(xa, ya), xyB=(xb, yb), coordsA="data", coordsB="data",
                                               axesA=ax, axesB=ax, color="black", alpha=alpha, linewidth=0.5, zorder=1)
                ax.add_patch(line)

    # Add Title
    plt.suptitle("Neural Network Architecture", fontsize=18, fontweight='bold', y=0.98)
    ax.text(0.5, 0.02, "Architecture: 3-Layer MLP | Activation: ReLU | Optimizer: Adam | Loss: Cross-Entropy", 
            ha='center', fontsize=12, style='italic', color='gray')

    # Save
    plt.tight_layout()
    output_file = 'FigA_NeuralNetwork_Architecture.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved as: {output_file}")
    plt.show()

if __name__ == "__main__":
    draw_neural_network_digital_twin()