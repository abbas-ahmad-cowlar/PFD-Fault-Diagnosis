"""
Exact Replica of Random Forest Conceptual Diagram
Precise visual matching to reference image
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
import numpy as np

def generate_rf_exact_replica(output_filename='PFD_SVM_Results_Production/Fig16_v5_RF_Conceptual_Exact.png'):
    """
    Generate exact replica with precise visual matching
    """
    # Create figure with proper aspect ratio
    fig, ax = plt.subplots(figsize=(11, 13), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    ax.set_facecolor('white')
    
    # ========================================================================
    # 1. DATASET (Large Dark Purple Oval - Top Center, Modern Style)
    # ========================================================================
    dataset_center_x = 5
    dataset_center_y = 12.2
    dataset_oval = Ellipse((dataset_center_x, dataset_center_y), 3.0, 1.1, 
                          facecolor='#6B46C1',  # Modern purple
                          edgecolor='#1E1B4B', linewidth=3.0,  # Darker edge for contrast
                          zorder=3)
    ax.add_patch(dataset_oval)
    ax.text(dataset_center_x, dataset_center_y, 'Dataset', ha='center', va='center',
            fontsize=19, fontweight='bold', color='white',
            family='Arial', zorder=4)
    
    # ========================================================================
    # 2. RANDOM FOREST TREES (3 trees with exact spacing)
    # ========================================================================
    tree_x_positions = [1.7, 5.0, 8.3]  # Precise horizontal spacing
    tree_labels = ['Decision tree-1', 'Decision tree-2', 'Decision tree-395']
    result_colors = ['#228B22', '#4169E1', '#000000']  # Green, Blue, Black
    result_labels = ['Result-1', 'Result-2', 'Result-395']  # Consistent labeling
    
    result_positions = []
    
    for idx, (x_pos, label, result_color, result_label) in enumerate(zip(
            tree_x_positions, tree_labels, result_colors, result_labels)):
        
        # Tree label (centered above tree, positioned higher to avoid arrow overlap)
        label_y = 10.5  # Moved higher
        ax.text(x_pos, label_y, label, ha='center', va='center',
               fontsize=14, fontweight='bold', color='#111827',  # Darker text
               family='Arial', zorder=5, 
               bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
                        edgecolor='#E5E7EB', linewidth=1, alpha=1.0))  # Better contrast background
        
        # ====================================================================
        # Draw decision tree structure (precise visual structure)
        # ====================================================================
        # Root node position
        root_y = 9.6
        root_radius = 0.24
        
        # Root node (large green circle, modern style, better contrast)
        root = Circle((x_pos, root_y), root_radius,
                     facecolor='#10B981',  # Modern green
                     edgecolor='#047857', linewidth=2.8,  # Darker edge
                     zorder=4)
        ax.add_patch(root)
        
        # First level: 2 blue nodes
        branch_y = root_y - 0.5
        branch_spacing = 0.4
        branch1_x = x_pos - branch_spacing
        branch2_x = x_pos + branch_spacing
        
        # Branch lines (smoother, modern style, better contrast)
        ax.plot([x_pos, branch1_x], [root_y - root_radius*0.7, branch_y + 0.16], 
               color='#1F2937', linewidth=2.5, zorder=3)  # Darker, thicker
        ax.plot([x_pos, branch2_x], [root_y - root_radius*0.7, branch_y + 0.16], 
               color='#1F2937', linewidth=2.5, zorder=3)  # Darker, thicker
        
        # First level nodes (blue circles, medium size, modern style, better contrast)
        node1 = Circle((branch1_x, branch_y), 0.17,
                      facecolor='#3B82F6',  # Modern blue
                      edgecolor='#1E40AF', linewidth=2.5,  # Darker edge
                      zorder=4)
        node2 = Circle((branch2_x, branch_y), 0.17,
                      facecolor='#3B82F6',  # Modern blue
                      edgecolor='#1E40AF', linewidth=2.5,  # Darker edge
                      zorder=4)
        ax.add_patch(node1)
        ax.add_patch(node2)
        
        # Second level: 4 leaf nodes (specific color patterns, no overlap)
        leaf_y = branch_y - 0.5
        leaf_spacing = 0.35  # Increased spacing to prevent overlap
        
        # Leaf patterns matching reference (modern colors):
        # Tree 1: left two blue, right two green
        # Tree 2: alternating (green, blue, green, blue)
        # Tree N: mixed (blue, green, green, blue)
        leaf_patterns = [
            ['#3B82F6', '#3B82F6', '#10B981', '#10B981'],  # Tree 1 (modern blue/green)
            ['#10B981', '#3B82F6', '#10B981', '#3B82F6'],  # Tree 2
            ['#3B82F6', '#10B981', '#10B981', '#3B82F6']   # Tree N
        ]
        
        leaf_colors = leaf_patterns[idx]
        
        # Calculate leaf positions (left to right, spread out to avoid overlap)
        # Left branch leaves (from branch1)
        leaf1_x = branch1_x - leaf_spacing
        leaf2_x = branch1_x + leaf_spacing * 0.3  # Closer to center but not overlapping
        
        # Right branch leaves (from branch2)
        leaf3_x = branch2_x - leaf_spacing * 0.3  # Closer to center but not overlapping
        leaf4_x = branch2_x + leaf_spacing
        
        # Ensure no overlap by checking spacing between leaf2 and leaf3
        min_gap = 0.25  # Minimum gap between center leaves
        if leaf3_x - leaf2_x < min_gap:
            # Adjust to create proper gap
            center_point = (branch1_x + branch2_x) / 2
            leaf2_x = center_point - min_gap / 2
            leaf3_x = center_point + min_gap / 2
        
        # Draw branches to leaves (smoother, modern style, better contrast)
        ax.plot([branch1_x, leaf1_x], [branch_y - 0.17, leaf_y + 0.13], 
               color='#1F2937', linewidth=2.0, zorder=3)  # Darker, thicker
        ax.plot([branch1_x, leaf2_x], [branch_y - 0.17, leaf_y + 0.13], 
               color='#1F2937', linewidth=2.0, zorder=3)
        ax.plot([branch2_x, leaf3_x], [branch_y - 0.17, leaf_y + 0.13], 
               color='#1F2937', linewidth=2.0, zorder=3)
        ax.plot([branch2_x, leaf4_x], [branch_y - 0.17, leaf_y + 0.13], 
               color='#1F2937', linewidth=2.0, zorder=3)
        
        # Leaf nodes (smaller circles, no overlap)
        leaf_positions = [
            (leaf1_x, leaf_y, leaf_colors[0]),
            (leaf2_x, leaf_y, leaf_colors[1]),
            (leaf3_x, leaf_y, leaf_colors[2]),
            (leaf4_x, leaf_y, leaf_colors[3])
        ]
        
        for leaf_x, leaf_y_pos, leaf_color in leaf_positions:
            # Determine edge color based on leaf color (darker for better contrast)
            edge_color = '#1E40AF' if leaf_color == '#3B82F6' else '#047857'
            leaf = Circle((leaf_x, leaf_y_pos), 0.14,
                         facecolor=leaf_color,
                         edgecolor=edge_color, linewidth=2.0,  # Thicker edge
                         zorder=4)
            ax.add_patch(leaf)
        
        tree_bottom = leaf_y - 0.14
        
        # ====================================================================
        # RESULT BOXES (centered, properly sized, in front of arrows)
        # ====================================================================
        result_box_width = 1.4
        result_box_height = 0.7
        result_box_x = x_pos - result_box_width / 2
        result_box_y = 6.8
        
        # Modern result box colors with better contrast
        modern_result_colors = ['#10B981', '#3B82F6', '#111827']  # Modern green, blue, dark gray
        modern_edge_colors = ['#047857', '#1E40AF', '#000000']  # Darker edges
        
        result_box = FancyBboxPatch((result_box_x, result_box_y), 
                                    result_box_width, result_box_height,
                                   boxstyle="round,pad=0.12",
                                   facecolor=modern_result_colors[idx],
                                   edgecolor=modern_edge_colors[idx], linewidth=3.0,  # Thicker edge
                                   zorder=5)  # Higher zorder to be in front of arrows
        ax.add_patch(result_box)
        
        # Text color: white for all (high contrast)
        text_color = 'white'
        ax.text(x_pos, result_box_y + result_box_height/2, result_label,
               ha='center', va='center',
               fontsize=13, fontweight='bold', color=text_color,
               family='Arial', zorder=6)  # Higher zorder than box to appear in front
        
        result_top_y = result_box_y + result_box_height
        result_positions.append((x_pos, result_top_y))
        
        # ====================================================================
        # ARROWS: Dataset -> Decision tree (pointing well below label to avoid overlap)
        # ====================================================================
        arrow_start_y = dataset_center_y - 0.55
        arrow_end_y = label_y - 0.5  # Much lower to avoid overlapping label
        arrow1 = FancyArrowPatch((dataset_center_x, arrow_start_y), (x_pos, arrow_end_y),
                                arrowstyle='->', mutation_scale=30,
                                linewidth=2.8, color='#111827', zorder=1)  # Darker, thicker
        ax.add_patch(arrow1)
        
        # ====================================================================
        # ARROWS: Random forest -> Result (vertical, centered)
        # ====================================================================
        arrow_start_y_tree = tree_bottom
        arrow_end_y_result = result_box_y
        arrow2 = FancyArrowPatch((x_pos, arrow_start_y_tree), (x_pos, arrow_end_y_result),
                                arrowstyle='->', mutation_scale=26,
                                linewidth=2.6, color='#111827', zorder=1)  # Darker, thicker
        ax.add_patch(arrow2)
    
    # ========================================================================
    # 3. ELLIPSIS (Between Decision tree-2 and Decision tree-395)
    # ========================================================================
    ellipsis_x = (tree_x_positions[1] + tree_x_positions[2]) / 2
    ellipsis_y = label_y - 0.5  # Lower to avoid overlapping labels, match arrow end
    ax.text(ellipsis_x, ellipsis_y, '...', ha='center', va='center',
           fontsize=40, fontweight='bold', color='black',
           family='Arial', zorder=5,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='none', alpha=0.9))  # White background
    
    # Dashed grey arrow from dataset to ellipsis (modern style, better contrast)
    arrow_ellipsis = FancyArrowPatch((dataset_center_x, arrow_start_y), 
                                    (ellipsis_x, ellipsis_y),
                                arrowstyle='->', mutation_scale=28,
                                linewidth=2.2, color='#6B7280', zorder=1,  # Darker gray
                                linestyle='--', alpha=0.7)
    ax.add_patch(arrow_ellipsis)
    
    # ========================================================================
    # 3B. ELLIPSIS BETWEEN RESULTS (Between Result-2 and Result-395)
    # ========================================================================
    # Get result box positions
    result_ellipsis_x = (tree_x_positions[1] + tree_x_positions[2]) / 2
    result_ellipsis_y = result_positions[0][1] - 0.35  # Same Y level as result boxes
    ax.text(result_ellipsis_x, result_ellipsis_y, '...', ha='center', va='center',
           fontsize=40, fontweight='bold', color='black',
           family='Arial', zorder=5,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='none', alpha=0.9))  # White background
    
    # ========================================================================
    # 4. AVERAGE RESULTS (Large Orange rectangle, centered, modern style)
    # ========================================================================
    avg_box_width = 3.5
    avg_box_height = 0.8
    avg_box_x = dataset_center_x - avg_box_width / 2
    avg_box_y = 4.5
    avg_box = FancyBboxPatch((avg_box_x, avg_box_y), avg_box_width, avg_box_height,
                            boxstyle="round,pad=0.15",
                            facecolor='#F59E0B',  # Modern orange
                            edgecolor='#B45309', linewidth=3.0,  # Darker, thicker edge
                            zorder=3)
    ax.add_patch(avg_box)
    ax.text(dataset_center_x, avg_box_y + avg_box_height/2, 'Average results', 
           ha='center', va='center',
           fontsize=17, fontweight='bold', color='white',
           family='Arial', zorder=4)
    
    # ========================================================================
    # 5. ARROWS: Results -> Average results (converging to center)
    # ========================================================================
    avg_box_top_y = avg_box_y + avg_box_height
    for x_pos, y_pos in result_positions:
        # Arrows go behind result boxes (zorder=2) but in front of average box (zorder=3)
        # Make arrows end slightly above the box so arrowheads are visible
        arrow_end_y = avg_box_top_y + 0.05  # Slightly above to show arrowhead
        arrow3 = FancyArrowPatch((x_pos, y_pos), (dataset_center_x, arrow_end_y),
                                arrowstyle='->', mutation_scale=28,
                                linewidth=2.8, color='#111827', zorder=3.5)  # Darker, thicker
        ax.add_patch(arrow3)
    
    # ========================================================================
    # 6. FINAL RESULT (Text in a shape at bottom, centered)
    # ========================================================================
    final_result_y = 2.5
    final_result_width = 2.5
    final_result_height = 0.7
    final_result_x = dataset_center_x - final_result_width / 2
    
    # Add rectangle around Final result (modern style, better contrast)
    final_result_box = FancyBboxPatch((final_result_x, final_result_y - final_result_height/2), 
                                     final_result_width, final_result_height,
                                    boxstyle="round,pad=0.12",
                                    facecolor='#8B5CF6',  # Darker purple for better contrast
                                    edgecolor='#6D28D9', linewidth=3.0,  # Darker, thicker edge
                                    zorder=3)
    ax.add_patch(final_result_box)
    
    ax.text(dataset_center_x, final_result_y, 'Final result', 
           ha='center', va='center',
           fontsize=19, fontweight='bold', color='white',
           family='Arial', zorder=4)
    
    # ========================================================================
    # 7. ARROW: Average results -> Final result (vertical, centered)
    # ========================================================================
    arrow_start_y_avg = avg_box_y
    arrow_end_y_final = final_result_y + final_result_height/2  # Top of final result box
    arrow4 = FancyArrowPatch((dataset_center_x, arrow_start_y_avg), 
                            (dataset_center_x, arrow_end_y_final),
                            arrowstyle='->', mutation_scale=32,
                            linewidth=2.8, color='#111827', zorder=1)  # Darker, thicker
    ax.add_patch(arrow4)
    
    # ========================================================================
    # Save figure
    # ========================================================================
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f"Saved: {output_filename}")
    plt.close()
    
    return output_filename


if __name__ == "__main__":
    output_file = generate_rf_exact_replica()
    print(f"\nExact replica generated with precise visual matching!")
    print(f"Output: {output_file}")
