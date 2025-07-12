#!/usr/bin/env python3
"""
Debug Roboflow Access

This script helps debug Roboflow API access and project structure.
"""

import os
from roboflow import Roboflow

def debug_roboflow_access():
    """Debug Roboflow workspace and project access."""
    
    # Get API key
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("❌ ROBOFLOW_API_KEY not set")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        print("✅ Roboflow connection established")
        
        # Get workspace info
        workspace = rf.workspace()
        print(f"✅ Workspace: {workspace}")
        
        # List projects in workspace
        print("\n📂 Available projects:")
        try:
            # This might not work on all accounts, but let's try
            projects = workspace.list_projects()
            for project in projects:
                print(f"  - {project}")
        except Exception as e:
            print(f"  ❌ Cannot list projects: {e}")
            print("  💡 Try listing projects manually in Roboflow web interface")
        
        # Try to access the specific project with different parsing
        project_variations = [
            # Original full ID
            "container_code_detection_pretrain_faster-mrjxf-kxw1i",
            # Without last part
            "container_code_detection_pretrain_faster-mrjxf", 
            # Just the main name
            "container_code_detection_pretrain_faster",
            # Maybe it's different
            "container-code-detection-pretrain-faster"
        ]
        
        print(f"\n🔍 Trying different project name variations:")
        for project_name in project_variations:
            try:
                project = workspace.project(project_name)
                print(f"  ✅ Found project: {project_name}")
                
                # Try to get versions
                try:
                    versions = project.versions()
                    print(f"    📋 Available versions: {versions}")
                except Exception as e:
                    print(f"    ❌ Cannot list versions: {e}")
                
                break
                
            except Exception as e:
                print(f"  ❌ Project '{project_name}' not found: {e}")
        
        # Check if we can access the model directly
        print(f"\n🎯 Trying to access model with different version formats:")
        version_variations = ["kxw1i", "1", "latest"]
        
        for version in version_variations:
            try:
                model = workspace.project("container_code_detection_pretrain_faster").version(version).model
                print(f"  ✅ Model accessible with version: {version}")
                break
            except Exception as e:
                print(f"  ❌ Version '{version}' failed: {e}")
                
    except Exception as e:
        print(f"❌ Roboflow connection failed: {e}")
        print("💡 Check your API key and internet connection")

if __name__ == "__main__":
    debug_roboflow_access() 