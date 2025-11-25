#!/usr/bin/env python3
"""
Simplified PRD Generator for PoSiVa Projects
Single-file, no orchestrator complexity
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# Add CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def log(message):
    """Simple logging with file append"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Append to log file
    log_file = Path(__file__).parent / 'prd_generation.log'
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')

def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

def init_vectorstore(config):
    """Initialize vectorstore from sample PRDs (for General AI/ML)"""
    log("Initializing vectorstore...")
    persist_dir = Path(__file__).parent.parent / 'vectorstore'
    source_dir = Path(__file__).parent.parent / 'sample_prds'
    
    if persist_dir.exists():
        vs = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=OpenAIEmbeddings(openai_api_key=config['openai_api_key'])
        )
        log(f"✅ Loaded vectorstore from {persist_dir}")
    else:
        log(f"Building new vectorstore from {source_dir}...")
        loader = DirectoryLoader(str(source_dir), glob="*.md", loader_cls=TextLoader)
        docs = loader.load()
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vs = Chroma.from_documents(
            splits,
            OpenAIEmbeddings(openai_api_key=config['openai_api_key']),
            persist_directory=str(persist_dir)
        )
        log(f"✅ Built vectorstore: {len(splits)} chunks")
    return vs

def get_open_projects(project_list_file, limit=None):
    """Parse project list and get OPEN projects"""
    log(f"Reading project list: {project_list_file}")
    with open(project_list_file) as f:
        lines = f.readlines()
    
    open_projects = []
    for line in lines:
        if '| OPEN |' in line:
            # Extract filename from table: | # | filename.md | OPEN |
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                filename = parts[2]
                # Remove _PRD.md suffix to get project name
                project_name = filename.replace('_PRD.md', '').replace('.md', '')
                open_projects.append((parts[1], filename, project_name))
    
    log(f"Found {len(open_projects)} OPEN projects")
    if limit:
        open_projects = open_projects[:limit]
        log(f"Limited to first {limit} project(s)")
    
    return open_projects

def update_status(project_list_file, project_name, new_status):
    """Update project status in list"""
    with open(project_list_file) as f:
        content = f.read()
    
    # Find and replace status for this project
    import re
    pattern = f'(\\|\\s*\\d+\\s*\\|\\s*{re.escape(project_name)}.*?\\.md\\s*\\|\\s*)OPEN(\\s*\\|)'
    updated = re.sub(pattern, f'\\g<1>{new_status}\\g<2>', content)
    
    with open(project_list_file, 'w') as f:
        f.write(updated)
    
    log(f"✅ Updated {project_name} → {new_status}")

def generate_prd(project_name, config):
    """Generate single PRD using 3-agent pipeline"""
    log(f"🚀 Starting generation for: {project_name}")
    start_time = datetime.now()
    
    # Set API key
    os.environ['OPENAI_API_KEY'] = config['openai_api_key']
    
    # Create LLM
    llm = ChatOpenAI(
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        openai_api_key=config['openai_api_key']
    )
    
    # Create agents
    researcher = Agent(
        role='Domain Researcher',
        goal=f'Extract key technical concepts for {project_name}',
        backstory='Technical researcher who analyzes similar projects',
        llm=llm,
        verbose=False
    )
    
    writer = Agent(
        role='PRD Writer',
        goal=f'Write complete PRD for {project_name}',
        backstory='Expert PRD writer for General AI/ML projects',
        llm=llm,
        verbose=False
    )
    
    validator = Agent(
        role='Quality Validator',
        goal='Verify PRD completeness and quality',
        backstory='QA expert who validates PRD structure',
        llm=llm,
        verbose=False
    )
    
    # Create tasks
    research_task = Task(
        description=f"""Research {project_name} and provide BRIEF summary (max 500 words):
        - Key technical concepts
        - Similar patterns from vectorstore
        - General AI/ML terminology (NO semiconductor terms)
        
        DO NOT write the PRD - just provide context.""",
        agent=researcher,
        expected_output="Brief research summary"
    )
    
    write_task = Task(
        description=f"""Write comprehensive PRD for {project_name}.

🚫 CRITICAL: GENERAL AI/ML PROJECT - ZERO SEMICONDUCTOR CONTAMINATION
DO NOT MENTION: STDF, wafer, die, semiconductor, silicon, ATE, yield, fab, IDDQ, tester, post-silicon validation, oscilloscopes, logic analyzers, Teradyne, Advantest, shmoo plots, wafer maps

FOCUS ON: Machine learning, deep learning, NLP, computer vision, data science, model training, inference, APIs, cloud deployment, user interfaces

DOCUMENT STRUCTURE (22 sections):
# Product Requirements Document (PRD) / # `{project_name}`

**Project ID, Category, Status, Version, Last Updated metadata**

## 1. Overview
### 1.1 Executive Summary
### 1.2 Document Purpose
### 1.3 Product Vision

## 2. Problem Statement
### 2.1 Current Challenges
### 2.2 Impact Analysis
### 2.3 Opportunity

## 3. Goals and Objectives
### 3.1 Primary Goals
### 3.2 Business Objectives
### 3.3 Success Metrics

## 4. Target Users/Audience
### 4.1 Primary Users
### 4.2 Secondary Users
### 4.3 User Personas (3+ detailed profiles with backgrounds, pain points, goals)

## 5. User Stories
(US-001, US-002... with As a/I want/So that/Acceptance Criteria)

## 6. Functional Requirements
### 6.1 Core Features (FR-001, FR-002...)
### 6.2 Advanced Features

## 7. Non-Functional Requirements
### 7.1 Performance
### 7.2 Reliability
### 7.3 Usability
### 7.4 Maintainability

## 8. Technical Requirements
### 8.1 Technical Stack (specific versions: Python 3.11+, React 18+, etc.)
### 8.2 AI/ML Components

## 9. System Architecture
### 9.1 High-Level Architecture (ASCII diagram)
### 9.2 Component Details
### 9.3 Data Flow

## 10. Data Model
### 10.1 Entity Relationships
### 10.2 Database Schema
### 10.3 Data Flow Diagrams
### 10.4 Input Data & Dataset Requirements

## 11. API Specifications
### 11.1 REST Endpoints
### 11.2 Request/Response Examples
### 11.3 Authentication

## 12. UI/UX Requirements
### 12.1 User Interface
### 12.2 User Experience
### 12.3 Accessibility

## 13. Security Requirements
### 13.1 Authentication
### 13.2 Authorization
### 13.3 Data Protection
### 13.4 Compliance

## 14. Performance Requirements
### 14.1 Response Times
### 14.2 Throughput
### 14.3 Resource Usage

## 15. Scalability Requirements
### 15.1 Horizontal Scaling
### 15.2 Vertical Scaling
### 15.3 Load Handling

## 16. Testing Strategy
### 16.1 Unit Testing
### 16.2 Integration Testing
### 16.3 Performance Testing
### 16.4 Security Testing

## 17. Deployment Strategy
### 17.1 Deployment Pipeline
### 17.2 Environments
### 17.3 Rollout Plan
### 17.4 Rollback Procedures

## 18. Monitoring & Observability
### 18.1 Metrics
### 18.2 Logging
### 18.3 Alerting
### 18.4 Dashboards

## 19. Risk Assessment
### 19.1 Technical Risks
### 19.2 Business Risks
### 19.3 Mitigation Strategies

## 20. Timeline & Milestones
### 20.1 Phase breakdown
### 20.2 Key Milestones

## 21. Success Metrics & KPIs
### 21.1 Measurable targets

## 22. Appendices & Glossary
### 22.1 Technical Background
### 22.2 References
### 22.3 Glossary (STDF, ATE, wafer, etc.)

Include:
- ASCII architecture diagrams
- Repository structure (notebooks/, src/, tests/, configs/, data/)
- Code snippets (API examples, config samples)
- Specific metrics (>90% accuracy, <500ms latency, 99.5% uptime)
- Detailed personas with names, backgrounds, pain points
- Realistic timelines and costs""",
        agent=writer,
        expected_output="Complete 22-section PRD",
        context=[research_task]
    )
    
    validate_task = Task(
        description=f"""Validate PRD for GENERAL AI/ML project {project_name}:
        
        CRITICAL CHECKS:
        1. ✅ All 22 sections present with proper headers
        2. ✅ Customized content (no generic placeholders)
        3. 🚫 ZERO SEMICONDUCTOR CONTAMINATION - Must NOT contain:
           - STDF, wafer, die, semiconductor, silicon (except in compound words)
           - ATE, yield (hardware), fab, IDDQ, tester
           - post-silicon validation, oscilloscopes, logic analyzers
           - Teradyne, Advantest, shmoo plots, wafer maps
        4. ✅ General AI/ML focus: LLM, NLP, ML, deep learning, APIs, cloud
        
        If ANY contamination found → VALIDATION FAILED - CONTAMINATED
        Otherwise → VALIDATION PASSED
        
        Output format:
        VALIDATION: [PASSED/FAILED]
        Sections: [count]/22
        Contamination: [CLEAN/CONTAMINATED - list terms found]
        Quality: [customized/generic]""",
        agent=validator,
        expected_output="Validation result",
        context=[write_task]
    )
    
    # Run crew (with tracing disabled)
    crew = Crew(
        agents=[researcher, writer, validator],
        tasks=[research_task, write_task, validate_task],
        process=Process.sequential,
        verbose=False
    )
    
    # Disable tracing to avoid prompts
    os.environ['CREWAI_TRACING_ENABLED'] = 'false'
    
    log("⏳ Running 3-agent pipeline (researcher → writer → validator)...")
    result = crew.kickoff()
    
    # Extract PRD content (from writer's output)
    if hasattr(result, 'tasks_output') and len(result.tasks_output) > 1:
        prd_content = result.tasks_output[1].raw
    else:
        prd_content = str(result)
    
    # Save PRD
    output_dir = Path(__file__).parent.parent / 'prd_files'
    output_file = output_dir / f"{project_name}_PRD.md"
    
    with open(output_file, 'w') as f:
        f.write(prd_content)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    lines = len(prd_content.split('\n'))
    words = len(prd_content.split())
    
    # POST-GENERATION CONTAMINATION CHECK (refined to avoid false positives)
    import subprocess
    # Use word boundaries (\b) for semiconductor-specific terms
    # Exclude: silicon (in compound words), tester (software QA), yield (general usage)
    contamination_check = subprocess.run(
        ['grep', '-iwE', '\\bSTDF\\b|\\bwafer\\b|post-silicon|\\bATE\\b|hardware.{0,20}yield|\\bdie\\b.{0,30}(yield|binning)|\\bfab\\b.{0,20}(process|manufacturing)|probe.{0,20}station|\\bIDDQ\\b|shmoo.{0,10}plot|Teradyne|Advantest',
         str(output_file)],
        capture_output=True, text=True
    )
    
    is_contaminated = contamination_check.returncode == 0
    
    if is_contaminated:
        # DELETE contaminated PRD
        output_file.unlink()
        log(f"❌ CONTAMINATION DETECTED - PRD deleted")
        matches = contamination_check.stdout.strip().split('\n')[:3]
        for match in matches:
            log(f"   ⚠️  {match[:100]}...")
        raise Exception(f"CONTAMINATED: {project_name}")
    
    log(f"✅ Generated: {output_file.name}")
    log(f"   📊 {lines} lines, ~{words} words | {elapsed:.1f}s")
    log(f"   🚫 Contamination: CLEAN")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate PoSiVa PRDs')
    parser.add_argument('--project-list', required=True, help='Path to posiva_project_list.md')
    parser.add_argument('--limit', type=int, default=None, help='Number of PRDs to generate')
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    log(f"🔧 Model: {config['llm']['model']} | Temp: {config['llm']['temperature']}")
    
    # Initialize vectorstore
    vectorstore = init_vectorstore(config)
    
    # Get OPEN projects
    project_list_path = Path(args.project_list)
    if not project_list_path.exists():
        log(f"❌ Project list not found: {project_list_path}")
        sys.exit(1)
    
    open_projects = get_open_projects(project_list_path, args.limit)
    
    if not open_projects:
        log("ℹ️  No OPEN projects found")
        return
    
    # Generate PRDs
    log(f"\n{'='*80}")
    log(f"Starting generation of {len(open_projects)} PRD(s)")
    log(f"{'='*80}\n")
    
    success_count = 0
    failed_count = 0
    summary = []
    
    for idx, (num, filename, project_name) in enumerate(open_projects, 1):
        log(f"\n📋 [{idx}/{len(open_projects)}] Project #{num}: {project_name}")
        log("-" * 80)
        
        # Check if PRD already exists (prevent duplicates)
        output_dir = Path(__file__).parent.parent / 'prd_files'
        potential_file = output_dir / f"{project_name}_PRD.md"
        if potential_file.exists():
            log(f"⚠️  PRD already exists! Skipping to avoid duplicate.")
            log(f"   File: {potential_file.name}")
            continue
        
        try:
            # Generate PRD
            output_file = generate_prd(project_name, config)
            
            # Update status to COMPLETED
            update_status(project_list_path, project_name, 'COMPLETED')
            
            success_count += 1
            summary.append({'name': project_name[:45], 'status': '✅ CLEAN'})
            
        except Exception as e:
            log(f"❌ Error: {e}")
            log(f"   Marking status as CONTAMINATED for user review")
            
            # Mark as CONTAMINATED so user can see it in the list
            update_status(project_list_path, project_name, 'CONTAMINATED')
            
            failed_count += 1
            summary.append({'name': project_name[:45], 'status': '❌ CONTAMINATED'})
            continue
    
    # VALIDATION: Check file count vs COMPLETED status count
    prd_dir = Path(__file__).parent.parent / 'prd_files'
    actual_prd_files = list(prd_dir.glob('*.md')) if prd_dir.exists() else []
    
    # Count COMPLETED in project list
    with open(project_list_path) as f:
        content = f.read()
    completed_in_list = content.count('| COMPLETED |')
    
    log(f"\n{'='*80}")
    log(f"🎉 Generation Complete!")
    log(f"   ✅ Successful: {success_count}/{len(open_projects)}")
    log(f"   ❌ Failed (contaminated): {failed_count}")
    log(f"   📁 PRDs saved to: prd_files/")
    
    # File validation check
    log(f"\n🔍 Validation Check:")
    log(f"   📝 COMPLETED in list: {completed_in_list}")
    log(f"   📄 PRD files on disk: {len(actual_prd_files)}")
    if completed_in_list == len(actual_prd_files):
        log(f"   ✅ MATCH - No duplicates detected")
    else:
        log(f"   ⚠️  MISMATCH - Possible duplicates or missing files!")
        log(f"      Difference: {abs(completed_in_list - len(actual_prd_files))} file(s)")
    
    if summary:
        log(f"\n📊 Summary Table:")
        log(f"{'-'*70}")
        log(f"{'Project':<50} {'Status':<20}")
        log(f"{'-'*70}")
        for item in summary:
            log(f"{item['name']:<50} {item['status']:<20}")
        log(f"{'-'*70}")
    
    # Separate contaminated list for user action
    contaminated = [item for item in summary if '❌' in item['status']]
    if contaminated:
        log(f"\n🚨 CONTAMINATED PRDs (require user action):")
        log(f"{'-'*70}")
        for idx, item in enumerate(contaminated, 1):
            log(f"   {idx}. {item['name']}")
        log(f"{'-'*70}")
        log(f"   Total contaminated: {len(contaminated)}")
        log(f"   Status: Marked as CONTAMINATED in project list")
        log(f"   Action: Manually change to OPEN to retry generation")
    
    log(f"{'='*80}\n")

if __name__ == '__main__':
    main()
