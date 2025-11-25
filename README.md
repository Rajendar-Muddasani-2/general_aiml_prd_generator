# General AI/ML PRD Generator 🤖

Automated PRD (Product Requirements Document) generation system for **General AI/ML** projects using AI agents powered by CrewAI and GPT-5.

## 🎯 Overview

This system generates comprehensive PRDs for general AI/ML projects including LLMs, NLP, computer vision, ML platforms, and AI applications. Each PRD includes 22 detailed sections covering everything from executive summary to deployment strategy.

🎉 **Status**: All 45 PRDs successfully generated!

### ✨ Key Features

- ✅ **Contamination Prevention**: Ensures PRDs contain NO semiconductor terminology
- ✅ **3-Agent Pipeline**: Research → Write → Validate workflow
- ✅ **Context-Aware Detection**: Distinguishes "software tester" from "hardware tester"
- ✅ **Duplicate Detection**: Skips generation if PRD already exists
- ✅ **Status Tracking**: Maintains OPEN/COMPLETED/CONTAMINATED states
- ✅ **Comprehensive Logging**: Full history in `prd_generation.log`
- ✅ **File Count Validation**: Ensures generated files match status records

## 📁 Repository Structure

```
general_aiml_prd_generator/
├── prd_files/                    # ✅ Generated PRD markdown files (45 COMPLETED)
├── sample_prds/                  # Clean sample PRDs for vectorstore
├── vectorstore/                  # Chroma vector database (auto-built from sample_prds/)
├── scripts/
│   ├── prd_generator_production.py   # 🎯 Main production script
│   ├── config.yaml               # Configuration (add your API key here)
│   └── prd_generation.log        # Execution logs
├── general_aiml_project_list.md  # Master project tracker (45 projects)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── .gitignore                    # Excludes API keys, logs, venv
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone repository
git clone <your-repo-url>
cd general_aiml_prd_generator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create `scripts/config.yaml` (ignored by git):

```yaml
openai_api_key: "sk-proj-YOUR_GPT5_KEY_HERE"

llm:
  model: "gpt-5"
  temperature: 0.7
```

⚠️ **IMPORTANT**: Never commit `config.yaml` - it's in `.gitignore`

### 3. Generate PRDs

#### Test Run (1 PRD)
```bash
cd scripts
../venv/bin/python prd_generator_production.py --project-list ../general_aiml_project_list.md --limit 1
```

#### Small Batch (5 PRDs)
```bash
../venv/bin/python prd_generator_production.py --project-list ../general_aiml_project_list.md --limit 5
```

#### Full Generation (All OPEN projects)
```bash
../venv/bin/python prd_generator_production.py --project-list ../general_aiml_project_list.md
```

Expected time: ~5-7 minutes per PRD

## 📊 Current Status

- **Total Projects**: 45
- **Completed**: 8 PRDs
- **Open**: 37 PRDs
- **Contaminated**: 0 (auto-deleted during generation)

Check `general_aiml_project_list.md` for detailed status.

## 🔍 How It Works

### 3-Agent Pipeline

1. **Researcher Agent**: Gathers domain context from vectorstore (clean AI/ML samples)
2. **Writer Agent**: Generates comprehensive 22-section PRD
3. **Validator Agent**: Ensures quality and NO semiconductor contamination

### Validation Layers

1. **Pre-Generation**: File existence check (prevents duplicates)
2. **During Generation**: CrewAI validator checks for required sections and general AI/ML focus
3. **Post-Generation**: `grep` check confirms NO semiconductor contamination
4. **End-of-Run**: File count validation (COMPLETED vs actual files)

### Contamination Detection (Refined)

PRDs **MUST NOT** contain semiconductor-specific terms:
- ❌ STDF, wafer, post-silicon
- ❌ ATE (Automated Test Equipment)
- ❌ Hardware yield, die yield, binning
- ❌ Fab processes, probe stations
- ❌ IDDQ, shmoo plots
- ❌ Teradyne, Advantest

✅ **Allowed general terms**:
- ✅ "tester" (software QA tester)
- ✅ "yield" (general performance yield)
- ✅ "silicon" in compound words (Silicon Valley)

**Context-aware detection** uses word boundaries to avoid false positives.

## 📝 Output Format

Each PRD includes 22 sections:

1. Executive Summary
2. Project Overview
3. Problem Statement
4. Goals & Objectives
5. Target Audience
6. User Stories & Use Cases
7. Functional Requirements
8. Technical Requirements
9. System Architecture
10. Data Model
11. API Specifications
12. UI/UX Requirements
13. Security Requirements
14. Performance Requirements
15. Scalability Requirements
16. Testing Strategy
17. Deployment Strategy
18. Monitoring & Maintenance
19. Documentation Requirements
20. Timeline & Milestones
21. Budget & Resources
22. Risks & Mitigation

Output file: `prd_files/<project_name>_PRD.md`

## 🛠️ Maintenance

### View Logs
```bash
tail -f scripts/prd_generation.log
```

### Check Status
```bash
grep -c "| COMPLETED |" general_aiml_project_list.md
ls prd_files/*.md | wc -l
```

### Retry Contaminated PRDs
1. Open `general_aiml_project_list.md`
2. Find projects with status `CONTAMINATED`
3. Manually change status to `OPEN`
4. Re-run generation

### Rebuild Vectorstore
```bash
rm -rf vectorstore/
# Will auto-rebuild on next run from sample_prds/
```

## 🔧 Configuration

Edit `scripts/config.yaml`:

```yaml
openai_api_key: "your-key-here"

llm:
  model: "gpt-5"        # or gpt-4o, gpt-4-turbo
  temperature: 0.7      # 0.0 (deterministic) to 1.0 (creative)
```

## 📈 Monitoring

The script provides real-time feedback:

```
[12:34:56] 🔧 Model: gpt-5 | Temp: 0.7
[12:34:56] ✅ Loaded vectorstore from vectorstore/
[12:34:56] Found 37 OPEN projects
[12:34:56] Starting generation of 5 PRD(s)
[12:35:02] 🚀 Starting generation for: aiml010_digital_twin_of_earth
[12:41:15] ✅ Generated: aiml010_digital_twin_of_earth_PRD.md
[12:41:15]    📊 721 lines, ~4106 words | 373.2s
[12:41:15]    🚫 Contamination: CLEAN
[12:41:15] ✅ Updated aiml010_digital_twin_of_earth → COMPLETED
```

## ⚠️ Troubleshooting

### "No module named 'crewai'"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Invalid API key"
Check `scripts/config.yaml` has correct key format: `sk-proj-...`

### "Contaminated PRD deleted"
PRD contained semiconductor-specific terms. System correctly rejected it to maintain general AI/ML focus.

### False Positive Detection
If general terms like "software tester" trigger contamination, the refined detection uses context-aware patterns. Update detection regex in script if needed.

### Tracing Prompts Appear
Environment variable `CREWAI_TRACING_ENABLED=false` is set in code. If still appearing, add to shell:
```bash
export CREWAI_TRACING_ENABLED=false
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Rajendar Muddasani**

## 🙏 Acknowledgments

- CrewAI for multi-agent orchestration
- OpenAI GPT-5 for content generation
- LangChain for vector store management
