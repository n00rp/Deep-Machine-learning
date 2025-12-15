#!/usr/bin/env python3
"""
Skapar nödvändiga mappar för outputs och säkerställer att projektet är redo att köras.
"""

import os

def create_output_directories():
    """Skapar alla nödvändiga output-mappar."""
    
    directories = [
        'outputs',
        'outputs/cam',
        'outputs/activation_max',
        'outputs/activation_max/features.5',
        'outputs/activation_max/features.10',
        'outputs/activation_max/features.20',
        'outputs/activation_max/features.28',
        'outputs/deep_dream',
        'outputs/progression'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Skapade mapp: {directory}")
    
    print("\n📁 Alla output-mappar är skapade!")
    print("Projektet är nu redo att köras.")

if __name__ == "__main__":
    create_output_directories()
