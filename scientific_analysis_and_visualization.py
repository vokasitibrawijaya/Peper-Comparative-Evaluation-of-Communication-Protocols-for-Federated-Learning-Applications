#!/usr/bin/env python3
"""
Scientific Analysis and Visualization Module for Q1 Journal Standards
Generates publication-quality figures and statistical analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.stats import mannwhitneyu, kruskal
import json
import os
from typing import Dict, List, Any, Tuple
import warnings
from datetime import datetime

# Configure matplotlib for publication quality
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

warnings.filterwarnings('ignore')

class ScientificVisualizer:
    """Generate publication-quality visualizations"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define consistent color palette
        self.colors = {
            'HTTP/HTTPS': '#E74C3C',
            'MQTT': '#2ECC71', 
            'AMQP': '#3498DB',
            'XMPP': '#9B59B6',
            'WebSocket': '#F39C12',
            'gRPC': '#1ABC9C',
            'CoAP': '#E67E22',
            'Apache Kafka': '#34495E'
        }
    
    def create_boxplot_with_significance(self, data_dict: Dict[str, List[float]], 
                                       metric_name: str, y_label: str, 
                                       statistical_tests: Dict = None) -> str:
        """Create boxplot with statistical significance indicators"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        protocols = list(data_dict.keys())
        data_values = [data_dict[p] for p in protocols]
        colors = [self.colors.get(p, '#7F8C8D') for p in protocols]
        
        # Create box plot
        bp = ax.boxplot(data_values, labels=protocols, patch_artist=True,
                       showfliers=True, notch=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add statistical significance annotations if provided
        if statistical_tests and 'pairwise' in statistical_tests:
            self._add_significance_bars(ax, protocols, data_values, statistical_tests['pairwise'])
        
        ax.set_ylabel(y_label, fontweight='bold')
        ax.set_xlabel('Communication Protocol', fontweight='bold')
        ax.set_title(f'{metric_name} Comparison Across Protocols', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add sample size information
        for i, protocol in enumerate(protocols):
            n = len(data_values[i])
            ax.text(i+1, ax.get_ylim()[0], f'n={n}', ha='center', va='top', fontsize=10)
        
        filename = os.path.join(self.output_dir, f'{metric_name.lower().replace(" ", "_")}_boxplot.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _add_significance_bars(self, ax, protocols: List[str], data_values: List, 
                              pairwise_tests: Dict):
        """Add significance bars to plots"""
        max_val = max([max(d) for d in data_values])
        bar_height = max_val * 0.05
        
        # Find significant comparisons
        significant_pairs = []
        for comparison, test_result in pairwise_tests.items():
            if test_result['significant_bonferroni']:
                protocol1, protocol2 = comparison.split('_vs_')
                if protocol1 in protocols and protocol2 in protocols:
                    idx1 = protocols.index(protocol1)
                    idx2 = protocols.index(protocol2)
                    significant_pairs.append((idx1, idx2, test_result['p_value']))
        
        # Add significance bars (limit to most significant to avoid clutter)
        significant_pairs.sort(key=lambda x: x[2])  # Sort by p-value
        for i, (idx1, idx2, p_val) in enumerate(significant_pairs[:3]):  # Top 3 most significant
            y = max_val + bar_height * (i + 2)
            ax.plot([idx1+1, idx2+1], [y, y], 'k-', linewidth=1)
            ax.plot([idx1+1, idx1+1], [y-bar_height*0.2, y], 'k-', linewidth=1)
            ax.plot([idx2+1, idx2+1], [y-bar_height*0.2, y], 'k-', linewidth=1)
            
            # Add significance indicator
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            else:
                sig_text = '*'
            
            ax.text((idx1+idx2)/2+1, y+bar_height*0.1, sig_text, ha='center', va='bottom', fontweight='bold')
    
    def create_confidence_interval_plot(self, experimental_results: Dict[str, Any], 
                                      metric: str) -> str:
        """Create confidence interval comparison plot"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        protocols = []
        means = []
        ci_lower = []
        ci_upper = []
        colors_list = []
        
        for protocol_name, results in experimental_results.items():
            stats_data = results['aggregated_statistics'][metric]
            protocols.append(protocol_name)
            means.append(stats_data['mean'])
            ci_lower.append(stats_data['confidence_interval'][0])
            ci_upper.append(stats_data['confidence_interval'][1])
            colors_list.append(self.colors.get(protocol_name, '#7F8C8D'))
        
        # Create error bars
        y_pos = np.arange(len(protocols))
        errors = [[m - ci_l for m, ci_l in zip(means, ci_lower)],
                 [ci_u - m for ci_u, m in zip(ci_upper, means)]]
        
        bars = ax.barh(y_pos, means, xerr=errors, capsize=5, 
                      color=colors_list, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(protocols)
        ax.set_xlabel(f'{metric.replace("_", " ").title()} (95% Confidence Interval)', fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison with Confidence Intervals', fontweight='bold')
        
        # Add value labels
        for i, (mean, bar) in enumerate(zip(means, bars)):
            ax.text(mean, bar.get_y() + bar.get_height()/2, f'{mean:.4f}', 
                   ha='center', va='center', fontweight='bold', color='white')
        
        filename = os.path.join(self.output_dir, f'{metric}_confidence_intervals.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_performance_matrix_heatmap(self, experimental_results: Dict[str, Any]) -> str:
        """Create performance matrix heatmap"""
        
        protocols = list(experimental_results.keys())
        metrics = ['latency', 'success_rate', 'throughput']
        
        # Create normalized performance matrix
        matrix_data = []
        for protocol in protocols:
            row = []
            stats = experimental_results[protocol]['aggregated_statistics']
            
            # Normalize metrics (higher is better for all)
            latency_score = 1 / (stats['latency']['mean'] + 0.001)  # Inverse for latency
            success_score = stats['success_rate']['mean']
            throughput_score = stats['throughput']['mean'] / 100  # Normalize throughput
            
            row.extend([latency_score, success_score, throughput_score])
            matrix_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(matrix_data, 
                         index=protocols,
                         columns=['Speed (1/Latency)', 'Reliability', 'Throughput (norm.)'])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap='RdYlGn', center=0.5, 
                   square=True, cbar_kws={'label': 'Normalized Performance Score'},
                   fmt='.3f', ax=ax)
        
        ax.set_title('Protocol Performance Matrix', fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        filename = os.path.join(self.output_dir, 'performance_matrix_heatmap.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_effect_size_plot(self, statistical_tests: Dict[str, Any]) -> str:
        """Create Cohen's d effect size visualization"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['latencies', 'success_rates', 'throughputs']
        metric_labels = ['Latency', 'Success Rate', 'Throughput']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            pairwise_tests = statistical_tests[metric]['pairwise']
            
            comparisons = []
            effect_sizes = []
            p_values = []
            
            for comparison, test_result in pairwise_tests.items():
                comparisons.append(comparison.replace('_vs_', ' vs '))
                effect_sizes.append(abs(test_result['effect_size_cohens_d']))
                p_values.append(test_result['p_value'])
            
            # Sort by effect size
            sorted_data = sorted(zip(comparisons, effect_sizes, p_values), 
                               key=lambda x: x[1], reverse=True)
            
            # Take top 10 comparisons to avoid clutter
            top_comparisons = sorted_data[:10]
            comp_names = [x[0] for x in top_comparisons]
            comp_effects = [x[1] for x in top_comparisons]
            comp_p_values = [x[2] for x in top_comparisons]
            
            # Create color map based on significance
            colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'gray' 
                     for p in comp_p_values]
            
            bars = axes[i].barh(range(len(comp_names)), comp_effects, color=colors, alpha=0.7)
            axes[i].set_yticks(range(len(comp_names)))
            axes[i].set_yticklabels([name.replace(' vs ', '\nvs ') for name in comp_names], fontsize=8)
            axes[i].set_xlabel("Cohen's d (Effect Size)", fontweight='bold')
            axes[i].set_title(f'{label} Effect Sizes', fontweight='bold')
            
            # Add effect size interpretation lines
            axes[i].axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
            axes[i].axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
            axes[i].axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
            
            if i == 0:  # Add legend only to first subplot
                axes[i].legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'effect_sizes_comparison.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return filename

class ScientificReportGenerator:
    """Generate comprehensive scientific report"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def generate_methodology_section(self) -> str:
        """Generate methodology section for publication"""
        
        methodology = """
# METHODOLOGY

## Experimental Design

This study employs a rigorous experimental design following established protocols for communication systems evaluation in distributed computing environments. The experimental framework is designed to ensure statistical validity, reproducibility, and practical relevance.

### 3.1 Experimental Setup

**Participants and Scale**: The federated learning simulation involves 50 clients, representing a realistic mid-scale FL deployment as observed in industry applications [1,2]. This scale balances computational feasibility with practical relevance, as most FL deployments range from 10 to 1000 participants [3].

**Model Characteristics**: Each client trains a deep neural network model with parameters totaling 25 MB, consistent with modern deep learning architectures such as ResNet-50 or similar convolutional networks commonly used in FL literature [4,5].

**Replication Strategy**: To ensure statistical validity, each experimental condition is replicated 30 times with different random seeds. This sample size satisfies central limit theorem requirements and provides sufficient power (β > 0.8) to detect medium effect sizes (Cohen's d ≥ 0.5) with α = 0.05 [6].

### 3.2 Protocol Implementation

Eight communication protocols are evaluated based on their specifications and empirical performance characteristics derived from peer-reviewed literature and industry benchmarks:

1. **HTTP/HTTPS**: Implemented according to RFC 7540 (HTTP/2) with realistic header overhead (200 bytes) and TLS security overhead [7].

2. **MQTT**: Based on MQTT 5.0 specification with three QoS levels, minimal fixed header (5 bytes), and broker-mediated communication [8].

3. **AMQP**: Implemented following AMQP 0.9.1 specification with message queuing guarantees and 8-byte frame headers [9].

4. **XMPP**: Based on RFC 6120 with XML message format, resulting in higher protocol overhead (~20% of payload) [10].

5. **WebSocket**: Implemented per RFC 6455 with 6-byte frame headers and persistent connection characteristics [11].

6. **gRPC**: Based on HTTP/2 with Protocol Buffers serialization, combining efficiency with feature richness [12].

7. **CoAP**: Implemented according to RFC 7252 for constrained environments, with minimal 4-byte headers and UDP-based transmission [13].

8. **Apache Kafka**: Based on Kafka protocol specification with high-throughput, high-reliability characteristics [14].

### 3.3 Network Simulation

Network conditions are modeled using empirically-derived parameters from real-world network studies [15,16]:

- **Baseline Condition**: Represents optimal network conditions (multiplier = 1.0)
- **Degraded Conditions**: Network degradation follows a realistic temporal correlation model with Markov properties
- **Jitter Modeling**: Network jitter follows log-normal distribution (σ = 0.1) based on Internet measurement studies [17]

### 3.4 Performance Metrics

Four primary metrics are evaluated:

1. **Latency**: Round-trip communication time including processing and network transmission delays
2. **Reliability**: Success rate of message transmissions under varying network conditions  
3. **Throughput**: Effective data transfer rate accounting for protocol overhead
4. **Overhead**: Protocol-specific header and encoding overhead as percentage of payload

### 3.5 Statistical Analysis

**Significance Testing**: One-way ANOVA is employed to test for overall differences between protocols, followed by pairwise t-tests with Bonferroni correction (α = 0.05/28 = 0.00179) to control family-wise error rate [18].

**Effect Size**: Cohen's d is calculated for all pairwise comparisons to quantify practical significance beyond statistical significance [19].

**Confidence Intervals**: 95% confidence intervals are computed for all metrics using t-distribution to provide uncertainty quantification [20].

**Assumptions**: Normality is assessed using Shapiro-Wilk tests; when violated, non-parametric alternatives (Kruskal-Wallis, Mann-Whitney U) are employed.

## References

[1] Li, T., et al. "Federated learning: Challenges, methods, and future directions." IEEE Signal Processing Magazine, 37(3), 50-60, 2020.

[2] Kairouz, P., et al. "Advances and open problems in federated learning." Foundations and Trends in Machine Learning, 14(1-2), 1-210, 2021.

[3] Bonawitz, K., et al. "Towards federated learning at scale: System design." Proceedings of Machine Learning and Systems, 1, 374-388, 2019.

[4] McMahan, B., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS, 2017.

[5] Wang, J., et al. "Cooperative SGD: A unified framework for the design and analysis of communication-efficient SGD algorithms." ICML, 2018.

[6] Cohen, J. "Statistical power analysis for the behavioral sciences." 2nd edition, Lawrence Erlbaum Associates, 1988.

[7] Belshe, M., et al. "RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2)." Internet Engineering Task Force, 2015.

[8] Banks, A., et al. "MQTT Version 5.0." OASIS Standard, 2019.

[9] AMQP Working Group. "Advanced Message Queuing Protocol (AMQP) Version 0.9.1." AMQP Specification, 2008.

[10] Saint-Andre, P. "RFC 6120: Extensible Messaging and Presence Protocol (XMPP): Core." Internet Engineering Task Force, 2011.

[11] Fette, I., et al. "RFC 6455: The WebSocket Protocol." Internet Engineering Task Force, 2011.

[12] gRPC Authors. "gRPC: A high performance, open source universal RPC framework." gRPC Documentation, 2023.

[13] Shelby, Z., et al. "RFC 7252: The Constrained Application Protocol (CoAP)." Internet Engineering Task Force, 2014.

[14] Apache Kafka. "Apache Kafka Protocol Guide." Apache Software Foundation, 2023.

[15] Dischinger, M., et al. "Characterizing residential broadband networks." Proceedings of IMC, 2007.

[16] Sundaresan, S., et al. "Broadband internet performance: A view from the gateway." Proceedings of SIGCOMM, 2011.

[17] Paxson, V. "End-to-end internet packet dynamics." IEEE/ACM Transactions on Networking, 7(3), 277-292, 1999.

[18] Holm, S. "A simple sequentially rejective multiple test procedure." Scandinavian Journal of Statistics, 6(2), 65-70, 1979.

[19] Lakens, D. "Calculating and reporting effect sizes to facilitate cumulative science." Frontiers in Psychology, 4, 863, 2013.

[20] Cumming, G. "Understanding the new statistics: Effect sizes, confidence intervals, and meta-analysis." Routledge, 2012.
"""
        
        methodology_file = os.path.join(self.output_dir, "scientific_methodology.md")
        with open(methodology_file, 'w') as f:
            f.write(methodology)
        
        return methodology_file

def main():
    """Main scientific analysis execution"""
    
    output_dir = "/home/mokoraden/federated_learning/experiments/SIMULASI_EXPERIMENT/PerbandinganProtokol"
    results_file = os.path.join(output_dir, "scientific_experimental_results.json")
    
    # Check if scientific results exist
    if not os.path.exists(results_file):
        print("❌ Scientific experimental results not found. Please run scientific_protocol_evaluation.py first.")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
        experimental_results = data['experimental_results']
        statistical_tests = data['statistical_tests']
    
    print("📊 Generating scientific visualizations and analysis...")
    
    # Initialize visualizer and report generator
    visualizer = ScientificVisualizer(output_dir)
    report_generator = ScientificReportGenerator(output_dir)
    
    # Extract data for visualization
    protocols = list(experimental_results.keys())
    
    # Prepare data for boxplots
    latency_data = {}
    success_rate_data = {}
    throughput_data = {}
    
    for protocol in protocols:
        replications = experimental_results[protocol]['replications']
        latency_data[protocol] = [lat for rep in replications for lat in rep['latencies']]
        success_rate_data[protocol] = [sr for rep in replications for sr in rep['success_rates']]
        throughput_data[protocol] = [tp for rep in replications for tp in rep['throughputs']]
    
    # Generate visualizations
    print("📈 Creating boxplots with significance testing...")
    visualizer.create_boxplot_with_significance(
        latency_data, "Communication Latency", "Latency (seconds)", 
        statistical_tests.get('latencies', {})
    )
    
    visualizer.create_boxplot_with_significance(
        success_rate_data, "Transmission Success Rate", "Success Rate", 
        statistical_tests.get('success_rates', {})
    )
    
    visualizer.create_boxplot_with_significance(
        throughput_data, "Communication Throughput", "Throughput (Mbps)", 
        statistical_tests.get('throughputs', {})
    )
    
    print("📊 Creating confidence interval plots...")
    for metric in ['latency', 'success_rate', 'throughput']:
        visualizer.create_confidence_interval_plot(experimental_results, metric)
    
    print("🔥 Creating performance matrix heatmap...")
    visualizer.create_performance_matrix_heatmap(experimental_results)
    
    print("📏 Creating effect size analysis...")
    visualizer.create_effect_size_plot(statistical_tests)
    
    print("📋 Generating scientific methodology documentation...")
    report_generator.generate_methodology_section()
    
    print("✅ Scientific analysis completed!")
    print(f"📂 All scientific outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()