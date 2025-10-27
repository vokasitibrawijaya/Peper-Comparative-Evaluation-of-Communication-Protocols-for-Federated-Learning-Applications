#!/usr/bin/env python3
"""
Analisis Lanjutan untuk Perbandingan Protokol Aplikasi
Menghasilkan heatmap, radar chart, comparison matrix, dan visualisasi advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Any
from datetime import datetime

# Set matplotlib backend untuk server
import matplotlib
matplotlib.use('Agg')

class ProtocolAnalyzer:
    """Advanced analysis tools for protocol comparison"""
    
    def __init__(self, output_dir: str, results_file: str = None):
        self.output_dir = output_dir
        self.results = None
        if results_file and os.path.exists(results_file):
            self.load_results(results_file)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def load_results(self, file_path: str):
        """Load results from JSON file"""
        with open(file_path, 'r') as f:
            self.results = json.load(f)
        print(f"📊 Loaded results from: {file_path}")
    
    def create_comparison_matrix(self) -> pd.DataFrame:
        """Create comparison matrix for all protocols"""
        if not self.results:
            print("❌ No results loaded!")
            return None
        
        data = []
        for protocol_name, result in self.results.items():
            data.append({
                'Protocol': protocol_name,
                'Avg_Latency_ms': result['avg_round_latency'] * 1000,
                'Success_Rate_%': result['success_rate'] * 100,
                'Overhead_%': result['avg_overhead_ratio'] * 100,
                'Total_Data_GB': result['total_data_gb'],
                'Failed_Transmissions': result['failed_transmissions'],
                'Successful_Rounds': result['successful_rounds']
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_file = os.path.join(self.output_dir, "protocol_comparison_matrix.csv")
        df.to_csv(csv_file, index=False)
        print(f"💾 Comparison matrix saved to: {csv_file}")
        
        return df
    
    def generate_heatmap_analysis(self):
        """Generate heatmap for protocol characteristics"""
        df = self.create_comparison_matrix()
        if df is None:
            return
        
        # Normalize data for heatmap (0-1 scale)
        normalized_data = df.copy()
        normalized_data['Latency_Score'] = 1 - (df['Avg_Latency_ms'] / df['Avg_Latency_ms'].max())
        normalized_data['Success_Score'] = df['Success_Rate_%'] / 100
        normalized_data['Efficiency_Score'] = 1 - (df['Overhead_%'] / df['Overhead_%'].max())
        
        # Select columns for heatmap
        heatmap_data = normalized_data[['Protocol', 'Latency_Score', 'Success_Score', 'Efficiency_Score']]
        heatmap_data = heatmap_data.set_index('Protocol')
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5,
                   cbar_kws={'label': 'Performance Score (0-1)'}, 
                   fmt='.3f', square=True)
        plt.title('Protocol Performance Heatmap', fontsize=16, fontweight='bold')
        plt.ylabel('Protocols', fontsize=12)
        plt.xlabel('Performance Metrics', fontsize=12)
        plt.tight_layout()
        
        heatmap_file = os.path.join(self.output_dir, "protocol_heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🔥 Heatmap saved to: {heatmap_file}")
    
    def radar_chart_comparison(self, selected_protocols: List[str] = None):
        """Create radar chart for protocol comparison"""
        if not self.results:
            return
        
        if selected_protocols is None:
            # Select top 4 protocols by performance
            performance_scores = {}
            for protocol_name, result in self.results.items():
                latency_score = 1.0 / (result['avg_round_latency'] + 0.01)
                success_score = result['success_rate']
                efficiency_score = 1.0 / (result['avg_overhead_ratio'] + 0.01)
                performance_scores[protocol_name] = (latency_score + success_score + efficiency_score) / 3
            
            selected_protocols = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)[:4]
            selected_protocols = [p[0] for p in selected_protocols]
        
        # Prepare data
        categories = ['Latency\n(Lower Better)', 'Reliability\n(Higher Better)', 
                     'Efficiency\n(Lower Overhead)', 'Data Transfer\n(Lower Better)']
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, protocol_name in enumerate(selected_protocols):
            if protocol_name not in self.results:
                continue
                
            result = self.results[protocol_name]
            
            # Normalize values (0-1 scale, higher is better)
            values = [
                1 - min(result['avg_round_latency'] / 0.5, 1.0),  # Latency (inverted)
                result['success_rate'],  # Reliability
                1 - min(result['avg_overhead_ratio'] / 0.3, 1.0),  # Efficiency (inverted)
                1 - min(result['total_data_gb'] / 50, 1.0)  # Data transfer (inverted)
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=3, label=protocol_name, 
                   color=colors[i % len(colors)], markersize=8)
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.set_title('Protocol Performance Radar Chart', size=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        radar_file = os.path.join(self.output_dir, "protocol_radar_chart.png")
        plt.savefig(radar_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📡 Radar chart saved to: {radar_file}")
    
    def generate_statistical_analysis(self):
        """Generate statistical analysis and correlation matrix"""
        if not self.results:
            return
        
        # Create dataframe for statistical analysis
        stats_data = []
        for protocol_name, result in self.results.items():
            # Calculate additional metrics
            reliability_score = result['success_rate'] * 100
            efficiency_score = (1 - result['avg_overhead_ratio']) * 100
            speed_score = (1 / result['avg_round_latency']) * 10  # Normalized speed
            
            stats_data.append({
                'Protocol': protocol_name,
                'Latency_ms': result['avg_round_latency'] * 1000,
                'Reliability_Score': reliability_score,
                'Efficiency_Score': efficiency_score,
                'Speed_Score': speed_score,
                'Data_GB': result['total_data_gb'],
                'Failed_Count': result['failed_transmissions']
            })
        
        df = pd.DataFrame(stats_data)
        
        # Generate correlation matrix
        numeric_cols = ['Latency_ms', 'Reliability_Score', 'Efficiency_Score', 
                       'Speed_Score', 'Data_GB', 'Failed_Count']
        corr_matrix = df[numeric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Protocol Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        corr_file = os.path.join(self.output_dir, "correlation_matrix.png")
        plt.savefig(corr_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Correlation matrix saved to: {corr_file}")
        
        # Generate statistical summary
        stats_summary = df[numeric_cols].describe()
        stats_file = os.path.join(self.output_dir, "statistical_summary.csv")
        stats_summary.to_csv(stats_file)
        print(f"📊 Statistical summary saved to: {stats_file}")
        
        return df, corr_matrix
    
    def generate_performance_trends(self):
        """Generate performance trends across rounds"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract round-by-round data
        protocols = list(self.results.keys())
        
        # Latency trends
        ax1 = axes[0, 0]
        for protocol_name in protocols:
            round_details = self.results[protocol_name]['round_details']
            rounds = [r['round'] for r in round_details]
            latencies = [r['latency'] for r in round_details]
            ax1.plot(rounds, latencies, marker='o', label=protocol_name, linewidth=2)
        
        ax1.set_title('Latency Trends Across Rounds', fontweight='bold')
        ax1.set_xlabel('Round Number')
        ax1.set_ylabel('Latency (seconds)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Data sent trends
        ax2 = axes[0, 1]
        for protocol_name in protocols:
            round_details = self.results[protocol_name]['round_details']
            rounds = [r['round'] for r in round_details]
            data_sent = [r['data_sent'] / (1024**3) for r in round_details]  # Convert to GB
            ax2.plot(rounds, data_sent, marker='s', label=protocol_name, linewidth=2)
        
        ax2.set_title('Data Transfer Trends Across Rounds', fontweight='bold')
        ax2.set_xlabel('Round Number')
        ax2.set_ylabel('Data Sent (GB)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Failure trends
        ax3 = axes[1, 0]
        for protocol_name in protocols:
            round_details = self.results[protocol_name]['round_details']
            rounds = [r['round'] for r in round_details]
            failures = [r['failures'] for r in round_details]
            ax3.plot(rounds, failures, marker='^', label=protocol_name, linewidth=2)
        
        ax3.set_title('Transmission Failures Across Rounds', fontweight='bold')
        ax3.set_xlabel('Round Number')
        ax3.set_ylabel('Failed Transmissions')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Network condition impact
        ax4 = axes[1, 1]
        network_conditions = ['Good (1.0)', 'Moderate (1.5)', 'Poor (2.0)']
        
        for protocol_name in protocols:
            round_details = self.results[protocol_name]['round_details']
            
            good_avg = np.mean([r['latency'] for r in round_details if r['network_condition'] == 1.0])
            moderate_avg = np.mean([r['latency'] for r in round_details if r['network_condition'] == 1.5])
            poor_avg = np.mean([r['latency'] for r in round_details if r['network_condition'] == 2.0])
            
            ax4.plot(network_conditions, [good_avg, moderate_avg, poor_avg], 
                    marker='D', label=protocol_name, linewidth=2, markersize=8)
        
        ax4.set_title('Network Condition Impact', fontweight='bold')
        ax4.set_xlabel('Network Condition')
        ax4.set_ylabel('Average Latency (seconds)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        trends_file = os.path.join(self.output_dir, "performance_trends.png")
        plt.savefig(trends_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Performance trends saved to: {trends_file}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        if not self.results:
            return
        
        report_lines = []
        report_lines.append("="*100)
        report_lines.append("📊 COMPREHENSIVE PROTOCOL ANALYSIS REPORT")
        report_lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📂 Output Directory: {self.output_dir}")
        report_lines.append("="*100)
        
        # Executive Summary
        report_lines.append("\n🎯 EXECUTIVE SUMMARY")
        report_lines.append("-" * 50)
        
        # Find best performers
        best_latency = min(self.results.items(), key=lambda x: x[1]['avg_round_latency'])
        best_reliability = max(self.results.items(), key=lambda x: x[1]['success_rate'])
        best_efficiency = min(self.results.items(), key=lambda x: x[1]['avg_overhead_ratio'])
        
        report_lines.append(f"⚡ Best Latency: {best_latency[0]} ({best_latency[1]['avg_round_latency']*1000:.1f}ms)")
        report_lines.append(f"🛡️  Best Reliability: {best_reliability[0]} ({best_reliability[1]['success_rate']*100:.1f}%)")
        report_lines.append(f"🗜️  Best Efficiency: {best_efficiency[0]} ({best_efficiency[1]['avg_overhead_ratio']*100:.1f}% overhead)")
        
        # Detailed Analysis
        report_lines.append("\n📋 DETAILED PERFORMANCE ANALYSIS")
        report_lines.append("-" * 50)
        
        for protocol_name, result in self.results.items():
            report_lines.append(f"\n🔍 {protocol_name}:")
            report_lines.append(f"   • Average Latency: {result['avg_round_latency']*1000:.1f}ms")
            report_lines.append(f"   • Success Rate: {result['success_rate']*100:.1f}%")
            report_lines.append(f"   • Protocol Overhead: {result['avg_overhead_ratio']*100:.1f}%")
            report_lines.append(f"   • Total Data Sent: {result['total_data_gb']:.2f} GB")
            report_lines.append(f"   • Failed Transmissions: {result['failed_transmissions']}")
            report_lines.append(f"   • Successful Rounds: {result['successful_rounds']}/12")
        
        # Recommendations
        report_lines.append("\n💡 PROTOCOL SELECTION GUIDE")
        report_lines.append("-" * 50)
        report_lines.append("🏃 For Real-time Applications: CoAP, MQTT")
        report_lines.append("🔒 For High-Reliability Systems: Apache Kafka, AMQP")
        report_lines.append("📱 For IoT/Edge Computing: CoAP, MQTT")
        report_lines.append("🌐 For Web-based Systems: WebSocket, HTTP/HTTPS")
        report_lines.append("📡 For Large-scale Deployments: Apache Kafka, gRPC")
        report_lines.append("⚖️  For Balanced Performance: gRPC, WebSocket")
        
        # Technical Insights
        report_lines.append("\n🔬 TECHNICAL INSIGHTS")
        report_lines.append("-" * 50)
        report_lines.append("• Lightweight protocols (CoAP, MQTT) excel in low-latency scenarios")
        report_lines.append("• Message queue protocols (Kafka, AMQP) provide superior reliability")
        report_lines.append("• Binary protocols (gRPC) offer good balance of speed and features")
        report_lines.append("• Text-based protocols (HTTP, XMPP) have higher overhead but better debugging")
        report_lines.append("• Network conditions significantly impact all protocols")
        
        report_text = "\n".join(report_lines)
        
        # Save comprehensive report
        report_file = os.path.join(self.output_dir, "comprehensive_analysis_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📋 Comprehensive report saved to: {report_file}")
        return report_text

def main():
    """Main analysis execution"""
    output_dir = "/home/mokoraden/federated_learning/experiments/SIMULASI_EXPERIMENT/PerbandinganProtokol"
    results_file = os.path.join(output_dir, "protocol_comparison_results.json")
    
    print(f"🔍 Starting Advanced Protocol Analysis")
    print(f"📂 Output Directory: {output_dir}")
    
    # Create analyzer
    analyzer = ProtocolAnalyzer(output_dir, results_file)
    
    if analyzer.results is None:
        print("❌ No results file found. Please run the main simulation first.")
        return
    
    print("\n📊 Generating advanced visualizations and analysis...")
    
    # Generate all analyses
    analyzer.create_comparison_matrix()
    analyzer.generate_heatmap_analysis()
    analyzer.radar_chart_comparison()
    analyzer.generate_statistical_analysis()
    analyzer.generate_performance_trends()
    analyzer.generate_comprehensive_report()
    
    print(f"\n✅ Advanced analysis completed!")
    print(f"📊 All charts and reports saved to: {output_dir}")

if __name__ == "__main__":
    main()