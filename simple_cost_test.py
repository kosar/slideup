"""
Simple test script to verify cost tracker functionality without requiring API keys.
"""

from podcast2video.cost_tracker import get_cost_tracker
import json

# Get the cost tracker instance
cost_tracker = get_cost_tracker()
cost_tracker.reset()

print("Running simple cost tracker test...")

# Add some test costs
print("Adding OpenAI chat costs...")
cost_tracker.add_openai_chat_cost(
    model="gpt-4",
    input_tokens=1000,
    output_tokens=500,
    operation_name="example_chat"
)

cost_tracker.add_openai_chat_cost(
    model="gpt-3.5-turbo",
    input_tokens=2000,
    output_tokens=1000,
    operation_name="example_chat_gpt35"
)

print("Adding OpenAI transcription cost...")
cost_tracker.add_openai_transcription_cost(
    duration_seconds=300,  # 5 minutes
    operation_name="example_transcription"
)

print("Adding Stability API image generation cost...")
cost_tracker.add_stability_image_cost(
    width=1024,
    height=1024,
    steps=50,
    samples=1,
    operation_name="example_image"
)

# Get the summary
summary = cost_tracker.get_summary()

# Print the results
print("\n=== Cost Tracker Test Results ===")
print(f"Total API cost: ${summary['total_cost']:.4f}")
print("\nCost breakdown:")
print(f"  OpenAI Chat: ${summary['api_breakdown']['openai']['chat']:.4f}")
print(f"  OpenAI Transcription: ${summary['api_breakdown']['openai']['transcription']:.4f}")
print(f"  Stability Image: ${summary['api_breakdown']['stability']['image']:.4f}")
print(f"\nNumber of entries: {summary['entry_count']}")

# Save report to file
report_path = "test_cost_report.json"
cost_tracker.save_report(report_path)
print(f"\nCost report saved to {report_path}")

# Display the detailed report content
with open(report_path, 'r') as f:
    report = json.load(f)

print("\nEntries in report:")
for i, entry in enumerate(report['entries']):
    print(f"  {i+1}. {entry['api_type']} - {entry['operation']}: ${entry['cost']:.4f}")

print("\nTest completed successfully!") 