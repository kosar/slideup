"""
End-to-end test for the cost tracker integration with the main application.
Tests the cost tracking functionality with actual API calls.

This test only makes minimal API calls to confirm the cost tracking
works properly with the actual APIs.
"""

import os
import sys
import json
import unittest
import tempfile
from unittest.mock import patch
from datetime import datetime
import time

# Add parent directory to path to find podcast2video modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from podcast2video.cost_tracker import get_cost_tracker
    from openai import OpenAI
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure to install the required packages: pip install -r requirements.txt")
    sys.exit(1)


class TestCostTrackerEndToEnd(unittest.TestCase):
    """Test the cost tracker integration with real API calls."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Check for API keys
        cls.openai_api_key = os.environ.get("OPENAI_API_KEY")
        cls.stability_api_key = os.environ.get("STABILITY_API_KEY")
        
        if not cls.openai_api_key:
            print("WARNING: OPENAI_API_KEY not found, OpenAI tests will be skipped")
        
        if not cls.stability_api_key:
            print("WARNING: STABILITY_API_KEY not found, Stability API tests will be skipped")
        
        # Get the cost tracker instance
        cls.cost_tracker = get_cost_tracker()
        cls.cost_tracker.reset()  # Reset to start clean
    
    def setUp(self):
        """Set up before each test."""
        # Reset the cost tracker before each test
        self.cost_tracker.reset()
    
    def test_openai_chat_completion_cost_tracking(self):
        """Test tracking cost for an OpenAI chat completion."""
        if not self.openai_api_key:
            self.skipTest("OPENAI_API_KEY not available")
        
        # Create OpenAI client
        client = OpenAI(api_key=self.openai_api_key)
        
        # Make a minimal API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello, World!' Please respond with only those words."}],
            max_tokens=10
        )
        
        # Track cost
        self.cost_tracker.add_openai_chat_cost(
            model="gpt-3.5-turbo",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            operation_name="test_chat"
        )
        
        # Check that cost was tracked
        summary = self.cost_tracker.get_summary()
        self.assertGreater(summary['total_cost'], 0)
        self.assertGreater(summary['api_breakdown']['openai']['chat'], 0)
        
        # Print cost info
        print(f"OpenAI Chat Cost: ${summary['api_breakdown']['openai']['chat']:.6f}")
        print(f"Input tokens: {response.usage.prompt_tokens}")
        print(f"Output tokens: {response.usage.completion_tokens}")
    
    def test_save_and_load_report(self):
        """Test saving a cost report and loading it back."""
        # Add a test cost
        self.cost_tracker.add_openai_chat_cost(
            model="gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            operation_name="test_save"
        )
        
        # Create a temporary file for the report
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            # Save the report
            self.cost_tracker.save_report(filepath)
            
            # Check that the file exists
            self.assertTrue(os.path.exists(filepath))
            
            # Load the report
            with open(filepath, 'r') as f:
                report = json.load(f)
            
            # Verify report content
            self.assertIn('total_cost', report)
            self.assertIn('api_breakdown', report)
            self.assertIn('entries', report)
            self.assertEqual(len(report['entries']), 1)
            
            # Verify entry content
            entry = report['entries'][0]
            self.assertEqual(entry['api_type'], 'openai')
            self.assertEqual(entry['operation'], 'test_save')
        
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        # Add several different costs
        self.cost_tracker.add_openai_chat_cost(
            model="gpt-4",
            input_tokens=200,
            output_tokens=100,
            operation_name="test_gpt4"
        )
        
        self.cost_tracker.add_openai_chat_cost(
            model="gpt-3.5-turbo",
            input_tokens=300,
            output_tokens=150,
            operation_name="test_gpt35"
        )
        
        self.cost_tracker.add_openai_transcription_cost(
            duration_seconds=120,
            operation_name="test_transcription"
        )
        
        self.cost_tracker.add_stability_image_cost(
            width=1024,
            height=1024,
            steps=50,
            operation_name="test_image"
        )
        
        # Get summary
        summary = self.cost_tracker.get_summary()
        
        # We should have 4 entries
        self.assertEqual(summary['entry_count'], 4)
        
        # We should have costs for each API type
        self.assertGreater(summary['api_breakdown']['openai']['chat'], 0)
        self.assertGreater(summary['api_breakdown']['openai']['transcription'], 0)
        self.assertGreater(summary['api_breakdown']['stability']['image'], 0)
        
        # Check that total is the sum of all costs
        total = (
            summary['api_breakdown']['openai']['chat'] +
            summary['api_breakdown']['openai']['transcription'] +
            summary['api_breakdown']['stability']['image']
        )
        self.assertAlmostEqual(summary['total_cost'], total, places=6)
        
        # Print the summary
        print("\nMultiple Operations Test Summary:")
        print(f"Total Cost: ${summary['total_cost']:.4f}")
        print(f"OpenAI Chat: ${summary['api_breakdown']['openai']['chat']:.4f}")
        print(f"OpenAI Transcription: ${summary['api_breakdown']['openai']['transcription']:.4f}")
        print(f"Stability Image: ${summary['api_breakdown']['stability']['image']:.4f}")
        print(f"Number of entries: {summary['entry_count']}")


if __name__ == '__main__':
    unittest.main() 