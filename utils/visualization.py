import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def generate_performance_dashboard(model, loader, correct_predictions, total_molecules):
    model.eval()
    all_preds = []
    all_trues = []
  
    with torch.no_grad():
        for batch in loader:
            _, pred_property, _, _ = model(batch.x, batch.edge_index, batch.batch)
            predictions = (pred_property >= 0.0).int().squeeze().cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            if predictions.ndim == 0:
                all_preds.append([predictions.item()])
                all_trues.append([labels.item()])
            else:
                all_preds.extend(predictions)
                all_trues.extend(labels)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    print("\tDETAILED CLASSIFICATION PERFORMANCE")
    print(classification_report(all_trues, all_preds, target_names=['Safe/Inactive', 'Active Tumour Fighter']))

    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Inactive', 'Predicted Active'],
                yticklabels=['True Inactive', 'True Active'])
    plt.title("Bio-Property Confusion Matrix Grid", fontsize=12, fontweight='bold')
    plt.ylabel("Ground Truth Lab Records")
    plt.xlabel("AI Predictive Decisions")

    plt.subplot(1, 2, 2)
    categories = ['Accurate Predictions', 'Misclassified Items']
    counts = [correct_predictions, total_molecules - correct_predictions]
    colors = ['#4CAF50', '#F44336']

    plt.bar(categories, counts, color=colors, width=0.5)
    for i, val in enumerate(counts):
        plt.text(i, val + 2, f"{val} ({val/total_molecules*100:.1f}%)", ha='center', fontweight='bold')

    plt.title("Overall Engineering Breakdown", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Molecules")
    plt.ylim(0, total_molecules + 20)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_dashboard.png', dpi=300) 
    plt.show()
