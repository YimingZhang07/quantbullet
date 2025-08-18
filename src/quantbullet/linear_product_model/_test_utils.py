import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantbullet.preprocessing.transformers import FlatRampTransformer
from quantbullet.dfutils import get_bins_and_labels


def generate_linear_product_example_data( n_samples=100_000, regression=True ):
    np.random.seed(42)
    x1 = np.random.uniform(0, 4, n_samples)
    x2 = np.random.uniform(4, 8, n_samples)
    y = ( x1 - 2 ) ** 2 + np.cos( 3 * x2 ) + np.random.normal(0, 1, n_samples) + 10

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

    x1_trans = FlatRampTransformer(
        knots = [0.5, 1, 1.5, 2, 2.5, 3, 3.5],
        include_bias=True
    )

    x2_trans = FlatRampTransformer(
        knots = [4.5, 5, 5.5, 6, 6.5, 7, 7.5],
        include_bias=True
    )

    train_df = np.concatenate([
        x1_trans.fit_transform(df['x1']),
        x2_trans.fit_transform(df['x2']),
    ], axis=1)

    train_df = pd.DataFrame(train_df, columns = x1_trans.get_feature_names_out().tolist() 
                            + x2_trans.get_feature_names_out().tolist())
    
    x1_bins, x1_labels = get_bins_and_labels(cutoffs=np.arange(0.2, 4, 0.2))
    x2_bins, x2_labels = get_bins_and_labels(cutoffs=np.arange(4.2, 8, 0.2))
    df['x1_bins'] = pd.cut( df['x1'], bins=x1_bins, labels=x1_labels )
    df['x2_bins'] = pd.cut( df['x2'], bins=x2_bins, labels=x2_labels )
    feature_groups = {'x1': x1_trans.get_feature_names_out().tolist(), 
                    'x2': x2_trans.get_feature_names_out().tolist(),}
    
    if regression:
        return df, train_df, feature_groups
    else:
        np.random.seed(15)
        probs = 1 / (1 + np.exp(-(df['y'] - 15)))
        df['binary_y'] = np.random.binomial(1, probs)
        df['binary_y'].mean()

        return df, train_df, feature_groups
    

def plot_predictions_by_bins(df, true_col='y', pred_col='model_predict'):
    for group in ['x1_bins', 'x2_bins']:
        plt.figure(figsize=(16, 3))
        summary = df.groupby(group, observed=True).agg({true_col: 'mean', pred_col: 'mean'})
        plt.plot(summary.index, summary[true_col], label='Actual', marker='o')
        plt.plot(summary.index, summary[pred_col], label='Predicted', marker='x')
        plt.title(f'Group: {group}')
        plt.xlabel(group)
        plt.ylabel('Mean Value')
        plt.legend()
