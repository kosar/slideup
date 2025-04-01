"""
Test suite for the API cost tracking module.

Run these tests with: pytest test_cost_tracker.py
"""

import unittest
import os
import json
import tempfile
from datetime import datetime
from podcast2video.cost_tracker import (
    CostTracker, CostEntry, OpenAICostConfig, StabilityCostConfig, get_cost_tracker
)

class TestCostEntry(unittest.TestCase):
    """Test the CostEntry class"""
    
    def test_cost_entry_creation(self):
        """Test creating a cost entry and converting to/from dict"""
        # Create a test entry
        entry = CostEntry(
            api_type='openai',
            operation='chat',
            cost=0.25,
            details={
                'model': 'gpt-4',
                'input_tokens': 500,
                'output_tokens': 300
            }
        )
        
        # Test properties
        self.assertEqual(entry.api_type, 'openai')
        self.assertEqual(entry.operation, 'chat')
        self.assertEqual(entry.cost, 0.25)
        self.assertEqual(entry.details['model'], 'gpt-4')
        
        # Test to_dict and from_dict
        entry_dict = entry.to_dict()
        self.assertEqual(entry_dict['api_type'], 'openai')
        self.assertEqual(entry_dict['cost'], 0.25)
        
        # Recreate from dict
        new_entry = CostEntry.from_dict(entry_dict)
        self.assertEqual(new_entry.api_type, entry.api_type)
        self.assertEqual(new_entry.operation, entry.operation)
        self.assertEqual(new_entry.cost, entry.cost)
        self.assertEqual(new_entry.details, entry.details)

class TestCostTracker(unittest.TestCase):
    """Test the CostTracker class"""
    
    def setUp(self):
        """Set up test environment"""
        self.tracker = CostTracker()
    
    def test_openai_chat_cost_gpt4(self):
        """Test calculating GPT-4 chat costs"""
        # Test case: 1000 input tokens, 500 output tokens with GPT-4
        cost = self.tracker.add_openai_chat_cost(
            model='gpt-4',
            input_tokens=1000,
            output_tokens=500,
            operation_name='test'
        )
        
        # Expected costs
        # Input: 1000 tokens / 1000 * $0.03 = $0.03
        # Output: 500 tokens / 1000 * $0.06 = $0.03
        # Total: $0.06
        expected_cost = 0.06
        
        self.assertAlmostEqual(cost, expected_cost, places=4)
        self.assertAlmostEqual(self.tracker.total_cost, expected_cost, places=4)
        self.assertAlmostEqual(self.tracker.api_totals['openai']['chat'], expected_cost, places=4)
        
        # Check that an entry was added
        self.assertEqual(len(self.tracker.entries), 1)
        entry = self.tracker.entries[0]
        self.assertEqual(entry.api_type, 'openai')
        self.assertEqual(entry.operation, 'test')
        self.assertAlmostEqual(entry.cost, expected_cost, places=4)
    
    def test_openai_chat_cost_gpt35(self):
        """Test calculating GPT-3.5 Turbo chat costs"""
        # Test case: 2000 input tokens, 1000 output tokens with GPT-3.5
        cost = self.tracker.add_openai_chat_cost(
            model='gpt-3.5-turbo',
            input_tokens=2000,
            output_tokens=1000,
            operation_name='test'
        )
        
        # Expected costs
        # Input: 2000 tokens / 1000 * $0.0015 = $0.003
        # Output: 1000 tokens / 1000 * $0.002 = $0.002
        # Total: $0.005
        expected_cost = 0.005
        
        self.assertAlmostEqual(cost, expected_cost, places=4)
        self.assertAlmostEqual(self.tracker.total_cost, expected_cost, places=4)
    
    def test_openai_transcription_cost(self):
        """Test calculating transcription costs"""
        # Test case: 5 minutes of audio
        duration_seconds = 300  # 5 minutes
        cost = self.tracker.add_openai_transcription_cost(
            duration_seconds=duration_seconds,
            operation_name='test_transcription'
        )
        
        # Expected cost: 5 minutes * $0.006 = $0.03
        expected_cost = 0.03
        
        self.assertAlmostEqual(cost, expected_cost, places=4)
        self.assertAlmostEqual(self.tracker.api_totals['openai']['transcription'], expected_cost, places=4)
    
    def test_stability_image_cost(self):
        """Test calculating Stability API image generation costs"""
        # Test case: 1024x1024 image with 50 steps
        cost = self.tracker.add_stability_image_cost(
            width=1024,
            height=1024,
            steps=50,
            samples=1,
            operation_name='test_image'
        )
        
        # Expected cost: $0.20 * 1 * 1 = $0.20
        expected_cost = 0.20
        
        self.assertAlmostEqual(cost, expected_cost, places=4)
        self.assertAlmostEqual(self.tracker.api_totals['stability']['image'], expected_cost, places=4)
        
        # Test multiple samples
        cost = self.tracker.add_stability_image_cost(
            width=768,
            height=768,
            steps=30,
            samples=3,
            operation_name='test_image_multi'
        )
        
        # Expected cost: $0.20 * 1 * 3 = $0.60
        expected_cost = 0.60
        
        self.assertAlmostEqual(cost, expected_cost, places=4)
    
    def test_get_summary(self):
        """Test getting cost summary"""
        # Add some test costs
        self.tracker.add_openai_chat_cost('gpt-4', 1000, 500, 'test1')
        self.tracker.add_openai_transcription_cost(300, 'whisper-1', 'test2')
        self.tracker.add_stability_image_cost(1024, 1024, 50, 1, 'test3')
        
        # Get the summary
        summary = self.tracker.get_summary()
        
        # Check summary contents
        self.assertIn('total_cost', summary)
        self.assertIn('start_time', summary)
        self.assertIn('api_breakdown', summary)
        self.assertIn('entry_count', summary)
        
        # Check that the total matches the sum of all operations
        expected_total = 0.06 + 0.03 + 0.20  # From the previous tests
        self.assertAlmostEqual(summary['total_cost'], expected_total, places=4)
        self.assertEqual(summary['entry_count'], 3)
    
    def test_get_detailed_report(self):
        """Test getting detailed cost report"""
        # Add a test cost
        self.tracker.add_openai_chat_cost('gpt-4', 1000, 500, 'test')
        
        # Get the detailed report
        report = self.tracker.get_detailed_report()
        
        # Check report contents
        self.assertIn('entries', report)
        self.assertEqual(len(report['entries']), 1)
        
        entry = report['entries'][0]
        self.assertEqual(entry['api_type'], 'openai')
        self.assertEqual(entry['operation'], 'test')
    
    def test_save_report(self):
        """Test saving report to file"""
        # Add a test cost
        self.tracker.add_openai_chat_cost('gpt-4', 1000, 500, 'test')
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            self.tracker.save_report(filepath)
            
            # Check that the file exists and contains valid JSON
            self.assertTrue(os.path.exists(filepath))
            
            with open(filepath, 'r') as f:
                report_data = json.load(f)
            
            self.assertIn('total_cost', report_data)
            self.assertIn('entries', report_data)
            self.assertEqual(len(report_data['entries']), 1)
        finally:
            # Clean up the temporary file
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_reset(self):
        """Test resetting the cost tracker"""
        # Add some test costs
        self.tracker.add_openai_chat_cost('gpt-4', 1000, 500, 'test1')
        self.tracker.add_stability_image_cost(1024, 1024, 50, 1, 'test2')
        
        # Verify costs were added
        self.assertEqual(len(self.tracker.entries), 2)
        self.assertGreater(self.tracker.total_cost, 0)
        
        # Reset the tracker
        self.tracker.reset()
        
        # Verify everything was reset
        self.assertEqual(len(self.tracker.entries), 0)
        self.assertEqual(self.tracker.total_cost, 0.0)
        self.assertEqual(self.tracker.api_totals['openai']['chat'], 0.0)
        self.assertEqual(self.tracker.api_totals['stability']['image'], 0.0)

class TestCostTrackerSingleton(unittest.TestCase):
    """Test the cost tracker singleton pattern"""
    
    def test_get_cost_tracker(self):
        """Test that get_cost_tracker returns the singleton instance"""
        tracker1 = get_cost_tracker()
        tracker2 = get_cost_tracker()
        
        # Both should be the same instance
        self.assertIs(tracker1, tracker2)
        
        # Add a cost to one and check it's reflected in the other
        tracker1.add_openai_chat_cost('gpt-4', 1000, 500, 'test')
        self.assertEqual(len(tracker2.entries), 1)

if __name__ == '__main__':
    unittest.main() 