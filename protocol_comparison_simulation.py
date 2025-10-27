#!/usr/bin/env python3
"""
Simulasi Perbandingan Protokol Aplikasi untuk Federated Learning
Membandingkan performa HTTP, MQTT, AMQP, XMPP, WebSocket, gRPC, CoAP, dan Apache Kafka
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# Set matplotlib backend untuk server
import matplotlib
matplotlib.use('Agg')

class ProtocolType(Enum):
    HTTP = "HTTP"
    MQTT = "MQTT"
    AMQP = "AMQP"
    XMPP = "XMPP"
    WEBSOCKET = "WebSocket"
    GRPC = "gRPC"
    COAP = "CoAP"
    KAFKA = "Apache Kafka"

@dataclass
class ProtocolCharacteristics:
    name: str
    overhead_ratio: float  # Protocol overhead as ratio of payload
    latency_base: float    # Base latency in seconds
    reliability: float     # 0-1 scale
    scalability: float     # 0-1 scale (higher = better)
    security_overhead: float  # Additional overhead for security
    qos_levels: int        # Number of QoS levels supported
    connection_type: str   # "persistent", "connectionless", "session-based"

class ApplicationProtocol:
    """Base class for application protocols in FL"""
    
    def __init__(self, characteristics: ProtocolCharacteristics):
        self.char = characteristics
        self.total_overhead = 0
        self.total_latency = 0
        self.successful_transmissions = 0
        self.failed_transmissions = 0
    
    def calculate_overhead(self, payload_size: int) -> int:
        """Calculate protocol overhead"""
        return int(payload_size * self.char.overhead_ratio)
    
    def calculate_latency(self, payload_size: int, network_condition: float = 1.0) -> float:
        """Calculate transmission latency"""
        base_latency = self.char.latency_base
        size_factor = payload_size / 1000000  # Per MB
        network_factor = network_condition  # 1.0 = good, 2.0 = poor
        
        return base_latency + (size_factor * 0.1) * network_factor
    
    def simulate_transmission(self, payload_size: int, qos_level: int = 0, 
                            network_condition: float = 1.0) -> Dict[str, Any]:
        """Simulate message transmission"""
        overhead = self.calculate_overhead(payload_size)
        total_size = payload_size + overhead
        latency = self.calculate_latency(total_size, network_condition)
        
        # Simulate transmission success based on reliability and network conditions
        success_probability = self.char.reliability * (1.0 / network_condition)
        success = random.random() < success_probability
        
        if success:
            self.successful_transmissions += 1
        else:
            self.failed_transmissions += 1
            latency *= 2  # Retry overhead
        
        return {
            'success': success,
            'latency': latency,
            'overhead': overhead,
            'total_size': total_size,
            'qos_level': qos_level
        }

class HTTPProtocol(ApplicationProtocol):
    """HTTP/HTTPS Protocol for FL"""
    
    def __init__(self):
        characteristics = ProtocolCharacteristics(
            name="HTTP/HTTPS",
            overhead_ratio=0.15,  # HTTP headers, JSON formatting
            latency_base=0.05,
            reliability=0.95,
            scalability=0.7,
            security_overhead=0.05,
            qos_levels=1,
            connection_type="connectionless"
        )
        super().__init__(characteristics)

class MQTTProtocol(ApplicationProtocol):
    """MQTT Protocol for FL"""
    
    def __init__(self):
        characteristics = ProtocolCharacteristics(
            name="MQTT",
            overhead_ratio=0.03,  # Very lightweight
            latency_base=0.02,
            reliability=0.9,
            scalability=0.95,
            security_overhead=0.02,
            qos_levels=3,
            connection_type="persistent"
        )
        super().__init__(characteristics)
    
    def simulate_transmission(self, payload_size: int, qos_level: int = 1, 
                            network_condition: float = 1.0) -> Dict[str, Any]:
        result = super().simulate_transmission(payload_size, qos_level, network_condition)
        
        # MQTT QoS handling
        if qos_level == 0:  # At most once
            result['latency'] *= 0.8
        elif qos_level == 1:  # At least once
            result['latency'] *= 1.0
        elif qos_level == 2:  # Exactly once
            result['latency'] *= 1.5
        
        return result

class AMQPProtocol(ApplicationProtocol):
    """AMQP Protocol for FL"""
    
    def __init__(self):
        characteristics = ProtocolCharacteristics(
            name="AMQP",
            overhead_ratio=0.08,
            latency_base=0.03,
            reliability=0.98,
            scalability=0.85,
            security_overhead=0.03,
            qos_levels=2,
            connection_type="persistent"
        )
        super().__init__(characteristics)

class XMPPProtocol(ApplicationProtocol):
    """XMPP Protocol for FL"""
    
    def __init__(self):
        characteristics = ProtocolCharacteristics(
            name="XMPP",
            overhead_ratio=0.25,  # XML overhead
            latency_base=0.04,
            reliability=0.92,
            scalability=0.6,
            security_overhead=0.08,
            qos_levels=1,
            connection_type="persistent"
        )
        super().__init__(characteristics)

class WebSocketProtocol(ApplicationProtocol):
    """WebSocket Protocol for FL"""
    
    def __init__(self):
        characteristics = ProtocolCharacteristics(
            name="WebSocket",
            overhead_ratio=0.05,
            latency_base=0.02,
            reliability=0.93,
            scalability=0.8,
            security_overhead=0.03,
            qos_levels=1,
            connection_type="persistent"
        )
        super().__init__(characteristics)

class gRPCProtocol(ApplicationProtocol):
    """gRPC Protocol for FL"""
    
    def __init__(self):
        characteristics = ProtocolCharacteristics(
            name="gRPC",
            overhead_ratio=0.06,
            latency_base=0.025,
            reliability=0.96,
            scalability=0.9,
            security_overhead=0.04,
            qos_levels=2,
            connection_type="persistent"
        )
        super().__init__(characteristics)

class CoAPProtocol(ApplicationProtocol):
    """CoAP Protocol for FL (IoT-focused)"""
    
    def __init__(self):
        characteristics = ProtocolCharacteristics(
            name="CoAP",
            overhead_ratio=0.02,  # Very lightweight for IoT
            latency_base=0.015,
            reliability=0.85,
            scalability=0.7,
            security_overhead=0.02,
            qos_levels=2,
            connection_type="connectionless"
        )
        super().__init__(characteristics)

class KafkaProtocol(ApplicationProtocol):
    """Apache Kafka Protocol for FL"""
    
    def __init__(self):
        characteristics = ProtocolCharacteristics(
            name="Apache Kafka",
            overhead_ratio=0.04,
            latency_base=0.08,  # Higher latency but high throughput
            reliability=0.99,
            scalability=0.98,
            security_overhead=0.05,
            qos_levels=3,
            connection_type="persistent"
        )
        super().__init__(characteristics)

class FederatedLearningProtocolSimulator:
    """Simulate FL with different application protocols"""
    
    def __init__(self, num_clients: int = 50, model_size: int = 10000000, output_dir: str = "."):
        self.num_clients = num_clients
        self.model_size = model_size  # 10MB model
        self.output_dir = output_dir
        self.protocols = [
            HTTPProtocol(),
            MQTTProtocol(),
            AMQPProtocol(),
            XMPPProtocol(),
            WebSocketProtocol(),
            gRPCProtocol(),
            CoAPProtocol(),
            KafkaProtocol()
        ]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def simulate_fl_round(self, protocol: ApplicationProtocol, num_rounds: int = 10,
                         network_conditions: List[float] = None) -> Dict[str, Any]:
        """Simulate FL training rounds with specific protocol"""
        
        if network_conditions is None:
            network_conditions = [1.0] * num_rounds  # Good network by default
        
        results = {
            'protocol_name': protocol.char.name,
            'total_latency': 0,
            'total_overhead': 0,
            'total_data_sent': 0,
            'successful_rounds': 0,
            'failed_transmissions': 0,
            'round_details': []
        }
        
        for round_num in range(num_rounds):
            round_latency = 0
            round_overhead = 0
            round_data = 0
            round_failures = 0
            
            network_condition = network_conditions[round_num]
            
            # Client to server communication (model updates)
            for client_id in range(self.num_clients):
                # Each client sends model update
                transmission = protocol.simulate_transmission(
                    payload_size=self.model_size,
                    qos_level=1,
                    network_condition=network_condition
                )
                
                if transmission['success']:
                    round_latency = max(round_latency, transmission['latency'])  # Synchronous FL
                    round_overhead += transmission['overhead']
                    round_data += transmission['total_size']
                else:
                    round_failures += 1
            
            # Server to client communication (global model broadcast)
            server_broadcast = protocol.simulate_transmission(
                payload_size=self.model_size,
                qos_level=1,
                network_condition=network_condition
            )
            
            if server_broadcast['success']:
                round_latency += server_broadcast['latency']
                round_overhead += server_broadcast['overhead'] * self.num_clients
                round_data += server_broadcast['total_size'] * self.num_clients
            
            # Record round results
            round_success = round_failures < (self.num_clients * 0.2)  # Allow 20% failures
            if round_success:
                results['successful_rounds'] += 1
            
            results['total_latency'] += round_latency
            results['total_overhead'] += round_overhead
            results['total_data_sent'] += round_data
            results['failed_transmissions'] += round_failures
            
            results['round_details'].append({
                'round': round_num + 1,
                'latency': round_latency,
                'overhead': round_overhead,
                'data_sent': round_data,
                'failures': round_failures,
                'network_condition': network_condition
            })
        
        # Calculate averages
        results['avg_round_latency'] = results['total_latency'] / num_rounds
        results['avg_overhead_ratio'] = results['total_overhead'] / results['total_data_sent'] if results['total_data_sent'] > 0 else 0
        results['success_rate'] = results['successful_rounds'] / num_rounds
        results['total_data_gb'] = results['total_data_sent'] / (1024**3)
        
        return results
    
    def run_comprehensive_comparison(self, num_rounds: int = 15) -> Dict[str, Any]:
        """Run comprehensive comparison across all protocols"""
        
        # Simulate varying network conditions
        network_conditions = []
        for i in range(num_rounds):
            if i < 5:
                network_conditions.append(1.0)  # Good network
            elif i < 10:
                network_conditions.append(1.5)  # Moderate network
            else:
                network_conditions.append(2.0)  # Poor network
        
        comparison_results = {}
        
        print("🌐 Federated Learning Application Protocol Comparison")
        print(f"👥 Clients: {self.num_clients}")
        print(f"📊 Model Size: {self.model_size / (1024*1024):.1f} MB")
        print(f"🔄 Rounds: {num_rounds}")
        print(f"📡 Network Conditions: Good→Moderate→Poor")
        print(f"💾 Output Directory: {self.output_dir}")
        print("=" * 70)
        
        for protocol in self.protocols:
            print(f"\n🔍 Testing {protocol.char.name}...")
            
            results = self.simulate_fl_round(protocol, num_rounds, network_conditions)
            comparison_results[protocol.char.name] = results
            
            # Print summary
            print(f"  ⏱️  Average Round Latency: {results['avg_round_latency']:.3f}s")
            print(f"  📈 Success Rate: {results['success_rate']*100:.1f}%")
            print(f"  📡 Total Data Sent: {results['total_data_gb']:.2f} GB")
            print(f"  🗜️  Overhead Ratio: {results['avg_overhead_ratio']*100:.1f}%")
            print(f"  ❌ Failed Transmissions: {results['failed_transmissions']}")
        
        return comparison_results
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed comparison report"""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("📋 DETAILED PROTOCOL ANALYSIS REPORT")
        report_lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        
        # Sort protocols by overall performance score
        performance_scores = {}
        for protocol_name, result in results.items():
            # Calculate composite performance score
            latency_score = 1.0 / (result['avg_round_latency'] + 0.01)  # Lower latency = better
            success_score = result['success_rate']
            efficiency_score = 1.0 / (result['avg_overhead_ratio'] + 0.01)  # Lower overhead = better
            
            performance_scores[protocol_name] = (latency_score + success_score + efficiency_score) / 3
        
        sorted_protocols = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        
        report_lines.append("\n🏆 PROTOCOL RANKING (Overall Performance):")
        for i, (protocol_name, score) in enumerate(sorted_protocols, 1):
            result = results[protocol_name]
            report_lines.append(f"{i:2d}. {protocol_name:15s} "
                              f"(Score: {score:.3f}, "
                              f"Latency: {result['avg_round_latency']:.3f}s, "
                              f"Success: {result['success_rate']*100:.1f}%, "
                              f"Overhead: {result['avg_overhead_ratio']*100:.1f}%)")
        
        # Category winners
        report_lines.append("\n🎯 CATEGORY WINNERS:")
        
        # Lowest latency
        best_latency = min(results.items(), key=lambda x: x[1]['avg_round_latency'])
        report_lines.append(f"⚡ Lowest Latency: {best_latency[0]} ({best_latency[1]['avg_round_latency']:.3f}s)")
        
        # Highest reliability
        best_reliability = max(results.items(), key=lambda x: x[1]['success_rate'])
        report_lines.append(f"🛡️  Highest Reliability: {best_reliability[0]} ({best_reliability[1]['success_rate']*100:.1f}%)")
        
        # Most efficient (lowest overhead)
        best_efficiency = min(results.items(), key=lambda x: x[1]['avg_overhead_ratio'])
        report_lines.append(f"🗜️  Most Efficient: {best_efficiency[0]} ({best_efficiency[1]['avg_overhead_ratio']*100:.1f}% overhead)")
        
        # Use case recommendations
        report_lines.append("\n💡 USE CASE RECOMMENDATIONS:")
        report_lines.append("🏃 Real-time FL (low latency): CoAP, MQTT, WebSocket")
        report_lines.append("🔒 High reliability FL: Apache Kafka, AMQP, gRPC")
        report_lines.append("📱 IoT/Edge FL: CoAP, MQTT")
        report_lines.append("🌐 Web-based FL: HTTP/HTTPS, WebSocket")
        report_lines.append("📡 Large-scale FL: Apache Kafka, MQTT, gRPC")
        
        report_text = "\n".join(report_lines)
        
        # Save report to file
        report_file = os.path.join(self.output_dir, "protocol_comparison_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text
    
    def plot_comparison_charts(self, results: Dict[str, Any]):
        """Generate comprehensive comparison charts"""
        
        protocols = list(results.keys())
        
        # Extract metrics
        latencies = [results[p]['avg_round_latency'] for p in protocols]
        success_rates = [results[p]['success_rate'] * 100 for p in protocols]
        overhead_ratios = [results[p]['avg_overhead_ratio'] * 100 for p in protocols]
        data_sent = [results[p]['total_data_gb'] for p in protocols]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                 '#FF8C94', '#98D8C8', '#A8E6CF']
        
        # 1. Average Round Latency
        bars1 = ax1.bar(protocols, latencies, color=colors[:len(protocols)])
        ax1.set_title('Average Round Latency Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Latency (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{latencies[i]:.3f}s', ha='center', va='bottom', fontsize=10)
        
        # 2. Success Rate
        bars2 = ax2.bar(protocols, success_rates, color=colors[:len(protocols)])
        ax2.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{success_rates[i]:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 3. Protocol Overhead
        bars3 = ax3.bar(protocols, overhead_ratios, color=colors[:len(protocols)])
        ax3.set_title('Protocol Overhead Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Overhead Ratio (%)')
        ax3.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{overhead_ratios[i]:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 4. Total Data Sent
        bars4 = ax4.bar(protocols, data_sent, color=colors[:len(protocols)])
        ax4.set_title('Total Data Transmitted', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Data (GB)')
        ax4.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{data_sent[i]:.2f}GB', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        chart_file = os.path.join(self.output_dir, "protocol_comparison_charts.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Network condition impact analysis
        self.plot_network_impact(results)
    
    def plot_network_impact(self, results: Dict[str, Any]):
        """Plot how protocols perform under different network conditions"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Group data by network conditions
        conditions = ['Good (1.0x)', 'Moderate (1.5x)', 'Poor (2.0x)']
        
        for protocol_name, result in results.items():
            round_details = result['round_details']
            
            # Group by network conditions
            good_latencies = [r['latency'] for r in round_details if r['network_condition'] == 1.0]
            moderate_latencies = [r['latency'] for r in round_details if r['network_condition'] == 1.5]
            poor_latencies = [r['latency'] for r in round_details if r['network_condition'] == 2.0]
            
            avg_latencies = [
                np.mean(good_latencies) if good_latencies else 0,
                np.mean(moderate_latencies) if moderate_latencies else 0,
                np.mean(poor_latencies) if poor_latencies else 0
            ]
            
            ax.plot(conditions, avg_latencies, marker='o', linewidth=2, 
                   label=protocol_name, markersize=8)
        
        ax.set_title('Protocol Performance Under Different Network Conditions', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Latency (seconds)')
        ax.set_xlabel('Network Condition')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        network_chart_file = os.path.join(self.output_dir, "network_impact_analysis.png")
        plt.savefig(network_chart_file, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main simulation execution"""
    # Set output directory
    output_dir = "/home/mokoraden/federated_learning/experiments/SIMULASI_EXPERIMENT/PerbandinganProtokol"
    
    # Create simulator
    simulator = FederatedLearningProtocolSimulator(
        num_clients=30, 
        model_size=5000000,  # 5MB model
        output_dir=output_dir
    )
    
    print(f"🚀 Starting Protocol Comparison Simulation")
    print(f"📂 Output Directory: {output_dir}")
    
    # Run comprehensive comparison
    results = simulator.run_comprehensive_comparison(num_rounds=12)
    
    # Generate detailed report
    simulator.generate_detailed_report(results)
    
    # Plot comparison charts
    simulator.plot_comparison_charts(results)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, "protocol_comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Simulation completed successfully!")
    print(f"💾 Results saved to: {results_file}")
    print(f"📊 Charts saved to: {output_dir}/")
    print(f"📋 Report saved to: {output_dir}/protocol_comparison_report.txt")

if __name__ == "__main__":
    main()