"""Simple cost tracker test."""

from podcast2video.cost_tracker import get_cost_tracker
import json

# Get the cost tracker instance
cost_tracker = get_cost_tracker()
cost_tracker.reset()

print("Running simple cost tracker test...")

# Add test costs
cost_tracker.add_openai_chat_cost(model="gpt-4", input_tokens=1000, output_tokens=500)
cost_tracker.add_openai_transcription_cost(duration_seconds=300)
cost_tracker.add_stability_image_cost(width=1024, height=1024)

# Get and print summary
summary = cost_tracker.get_summary()
print("\nTotal API cost: $", summary["total_cost"])
print("OpenAI Chat cost: $", summary["api_breakdown"]["openai"]["chat"])
print("OpenAI Transcription cost: $", summary["api_breakdown"]["openai"]["transcription"])
print("Stability Image cost: $", summary["api_breakdown"]["stability"]["image"])
print("Number of entries:", summary["entry_count"])

# Save report to file
report_path = "test_cost_report.json"
cost_tracker.save_report(report_path)
print("\nCost report saved to", report_path)
