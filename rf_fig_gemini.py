import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np

def draw_final_corrected_system(output_filename='Fig_Final_Corrected.png'):
    # ==========================================
    # 1. SETUP & STYLE
    # ==========================================
    # Increased figure width slightly to prevent overflow
    fig, ax = plt.subplots(figsize=(17, 9), dpi=300)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Professional Palette
    COLORS = {
        'input': '#2563EB',        # Royal Blue
        'card_bg': '#F8FAFC',      # Very light slate
        'card_border': '#CBD5E1',  # Light gray border
        'connector': '#64748B',    # Darker connector for visibility
        'tree_internal': '#64748B', # Darker gray for internal tree structure
        
        # Fault Classes
        'cavitation': '#DC2626',   # Red
        'lubrication': '#D97706',  # Amber
        'jeu': '#10B981',          # Emerald Green
        
        'vote_box': '#334155',     # Dark Slate
        'text_main': '#1E293B',
        'text_sub': '#475569'      # Darker sub-text
    }

    # ==========================================
    # 2. DRAWING PRIMITIVES
    # ==========================================
    def draw_smooth_connector(p1, p2, color, lw=1.5, style='solid'):
        """Draws a horizontal sigmoid curve"""
        x1, y1 = p1
        x2, y2 = p2
        dist = (x2 - x1) * 0.6
        
        verts = [(x1, y1), (x1 + dist, y1), (x2 - dist, y2), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color, lw=lw, linestyle=style, zorder=1)
        ax.add_patch(patch)

    def draw_tree_structure(ax, x, y, width, height, color, winner_idx=3):
        """
        Draws a mini binary tree.
        winner_idx: 0=l2_ll, 1=l2_lr, 2=l2_rl, 3=l2_rr
        """
        root = (x, y + height*0.35)
        l1_left = (x - width*0.25, y)
        l1_right = (x + width*0.25, y)
        l2_ll = (l1_left[0] - width*0.1, y - height*0.35)
        l2_lr = (l1_left[0] + width*0.1, y - height*0.35)
        l2_rl = (l1_right[0] - width*0.1, y - height*0.35)
        l2_rr = (l1_right[0] + width*0.1, y - height*0.35)
        
        leaves = [l2_ll, l2_lr, l2_rl, l2_rr]
        winning_leaf = leaves[winner_idx]

        # Draw Edges
        kw = dict(color=COLORS['tree_internal'], lw=2.0, zorder=15)
        # Root to L1
        ax.plot([root[0], l1_left[0]], [root[1], l1_left[1]], **kw)
        ax.plot([root[0], l1_right[0]], [root[1], l1_right[1]], **kw)
        # L1 to L2
        ax.plot([l1_left[0], l2_ll[0]], [l1_left[1], l2_ll[1]], **kw)
        ax.plot([l1_left[0], l2_lr[0]], [l1_left[1], l2_lr[1]], **kw)
        ax.plot([l1_right[0], l2_rl[0]], [l1_right[1], l2_rl[1]], **kw)
        ax.plot([l1_right[0], l2_rr[0]], [l1_right[1], l2_rr[1]], **kw)
        
        # Highlight winning path
        path_kw = dict(color=color, lw=2.0, zorder=16)
        if winner_idx <= 1: # Left branch winner
            ax.plot([root[0], l1_left[0]], [root[1], l1_left[1]], **path_kw)
            ax.plot([l1_left[0], winning_leaf[0]], [l1_left[1], winning_leaf[1]], **path_kw)
        else: # Right branch winner
            ax.plot([root[0], l1_right[0]], [root[1], l1_right[1]], **path_kw)
            ax.plot([l1_right[0], winning_leaf[0]], [l1_right[1], winning_leaf[1]], **path_kw)

        # Draw Nodes
        nodes = [root, l1_left, l1_right, l2_ll, l2_lr, l2_rl, l2_rr]
        for i, n in enumerate(nodes):
            # Skip drawing the winning leaf as a regular node
            is_winning_leaf = (i == 3 and winner_idx == 0) or \
                              (i == 4 and winner_idx == 1) or \
                              (i == 5 and winner_idx == 2) or \
                              (i == 6 and winner_idx == 3)
            if not is_winning_leaf:
                ax.scatter(n[0], n[1], s=50, c=COLORS['tree_internal'], zorder=20)
            
        # Winning Node
        ax.scatter(winning_leaf[0], winning_leaf[1], s=80, c=color, zorder=21, edgecolors='white', linewidth=1.5)
        
        return (x + width/2, y)

    # ==========================================
    # 3. BUILD ARCHITECTURE
    # ==========================================

    # --- A. DATA SOURCE ---
    data_x, data_y = 2.0, 5.0
    ax.add_patch(mpatches.FancyBboxPatch((data_x-1.5, data_y-0.8), 3.0, 1.6, 
                 boxstyle="round,pad=0.1", fc=COLORS['input'], ec='none', zorder=10))
    ax.text(data_x, data_y+0.2, "Vibration Signal", color='white', ha='center', fontweight='bold', fontsize=13, zorder=11)
    ax.text(data_x, data_y-0.3, "Time-Domain\nFeatures", color='white', ha='center', fontsize=10, zorder=11)

    # --- B. ENSEMBLE CONTAINER ---
    rect = mpatches.FancyBboxPatch((4.5, 0.5), 5.0, 9.0, boxstyle="round,pad=0.2", 
                                   fc='#FFFFFF', ec='#94A3B8', linestyle='--', lw=1.5, zorder=0)
    ax.add_patch(rect)
    ax.text(7.0, 9.8, "Ensemble (n=395 Trees)", ha='center', fontsize=13, fontweight='bold', color=COLORS['text_main'])

    # --- C. TREE CARDS ---
    # Added 'winner_idx' to vary the winning leaf position
    trees = [
        {'name': 'Tree 1',   'y': 8.5, 'pred': 'Cavitation',    'c': COLORS['cavitation'],  'winner_idx': 3}, # Rightmost
        {'name': 'Tree 2',   'y': 6.3, 'pred': 'Lubrification', 'c': COLORS['lubrication'], 'winner_idx': 1}, # Inner Left
        {'name': 'Tree 3',   'y': 4.1, 'pred': 'Jeu',           'c': COLORS['jeu'],         'winner_idx': 0}, # Leftmost
        {'name': 'Tree 395', 'y': 1.9, 'pred': 'Cavitation',    'c': COLORS['cavitation'],  'winner_idx': 3}, # Rightmost
    ]

    vote_x, vote_y = 12.0, 5.0 # Defined here for use in loop

    for i, t in enumerate(trees):
        cx, cy = 7.0, t['y']
        
        # Card
        card_w, card_h = 2.4, 1.4
        card = mpatches.FancyBboxPatch((cx-card_w/2, cy-card_h/2), card_w, card_h,
                                      boxstyle="round,pad=0.1", fc=COLORS['card_bg'], 
                                      ec=t['c'], lw=1.5, zorder=5)
        ax.add_patch(card)
        
        # Tree Structure with varied winner
        draw_tree_structure(ax, cx, cy, card_w*0.7, card_h*0.7, t['c'], t['winner_idx'])
        
        # Tree Name
        ax.text(cx, cy + card_h/2 + 0.15, t['name'], ha='center', fontsize=10, fontweight='bold', color=COLORS['text_sub'])

        # Input Connector
        input_anchor = (data_x + 1.5, data_y + (1.5-i)*0.4)
        card_left = (cx - card_w/2, cy)
        draw_smooth_connector(input_anchor, card_left, COLORS['connector'], lw=1.2)
        
        # Subset Label
        mid_x = (input_anchor[0] + card_left[0]) / 2
        mid_y = (input_anchor[1] + card_left[1]) / 2
        lbl = f"Subset {i+1}" if i < 3 else "Subset 395"
        ax.text(mid_x, mid_y, lbl, ha='center', va='center', fontsize=9, color=COLORS['text_sub'], 
                bbox=dict(fc='white', ec='none', pad=0.5))

        # Output Connector
        card_right = (cx + card_w/2, cy)
        vote_in_y = vote_y + (1.5-i)*0.5
        draw_smooth_connector(card_right, (vote_x - 1.6, vote_in_y), t['c'], lw=1.5)
        
        # Prediction Label
        txt_x = card_right[0] + 0.8
        txt_y = (cy + vote_in_y) / 2
        ax.text(txt_x, txt_y, t['pred'], fontsize=9, color=t['c'], fontweight='bold', ha='center',
               bbox=dict(fc='white', ec='none', pad=1, alpha=0.9))

    # Ellipsis
    ax.text(7.0, 3.0, "• • •", ha='center', fontsize=24, color=COLORS['text_main'], rotation=90)

    # --- D. MAJORITY VOTING ---
    vote_w, vote_h = 3.2, 2.0
    ax.add_patch(mpatches.FancyBboxPatch((vote_x-vote_w/2, vote_y-vote_h/2), vote_w, vote_h,
                 boxstyle="round,pad=0.1", fc=COLORS['vote_box'], ec='none', zorder=10))
    ax.text(vote_x, vote_y + 0.6, "Majority Voting", color='white', ha='center', fontweight='bold', fontsize=12, zorder=11)
    
    # Histogram - Corrected heights (2 Red, 1 Orange, 1 Green)
    ax.plot([vote_x-0.8, vote_x+0.8], [vote_y-0.5, vote_y-0.5], color='white', lw=1, zorder=11)
    ax.add_patch(mpatches.Rectangle((vote_x-0.7, vote_y-0.5), 0.4, 0.8, fc=COLORS['cavitation'], zorder=11)) # 2 votes
    ax.add_patch(mpatches.Rectangle((vote_x-0.2, vote_y-0.5), 0.4, 0.4, fc=COLORS['lubrication'], zorder=11)) # 1 vote
    ax.add_patch(mpatches.Rectangle((vote_x+0.3, vote_y-0.5), 0.4, 0.4, fc=COLORS['jeu'], zorder=11))         # 1 vote

    # --- E. FINAL RESULT ---
    # Shifted left slightly to fit within the new figure bounds
    final_x = 15.2 
    draw_smooth_connector((vote_x + vote_w/2, vote_y), (final_x - 1.2, vote_y), COLORS['vote_box'], lw=3)
    
    res_w, res_h = 2.4, 1.2
    ax.add_patch(mpatches.FancyBboxPatch((final_x-res_w/2, vote_y-res_h/2), res_w, res_h,
                 boxstyle="round,pad=0.1", fc='white', ec=COLORS['cavitation'], lw=3, zorder=10))
    
    ax.text(final_x, vote_y+0.25, "Final Diagnosis", color=COLORS['text_sub'], ha='center', fontsize=10, zorder=11)
    ax.text(final_x, vote_y-0.15, "CAVITATION", color=COLORS['cavitation'], ha='center', fontweight='bold', fontsize=13, zorder=11)

    # --- F. LEGEND ---
    # Adjusted position
    leg_x, leg_y = 15.0, 0.5
    ax.text(leg_x, leg_y+0.8, "Fault Classes:", ha='center', fontweight='bold', fontsize=11)
    classes = [("Cavitation", COLORS['cavitation']), ("Lubrification", COLORS['lubrication']), ("Jeu", COLORS['jeu'])]
    for i, (name, col) in enumerate(classes):
        ly = leg_y + 0.5 - (i*0.25)
        ax.scatter(leg_x - 0.9, ly, c=col, s=60)
        ax.text(leg_x - 0.7, ly, name, va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Generated Final Corrected Diagram: {output_filename}")
    plt.close()

if __name__ == "__main__":
    draw_final_corrected_system()