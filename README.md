# ascendc

An engineering Skills repository for the Ascend / CANN development ecosystem, used to accumulate high-frequency R&D assistance capabilities and support kernel development, performance optimization, NPU migration, and collaborative document updates.

## Project Overview

`ascendc` focuses on typical repetitive work in Ascend-related R&D workflows. Through reusable Skills, it turns experience into practical capabilities that are executable, collaborative, and maintainable.  
The repository currently covers the following directions:

- Kernel / operator basic scaffold generation
- Kernel / operator performance optimization analysis
- Migration assistance from CUDA/CPU to Ascend NPU
- Technical documentation and explanatory document updates

The goal is to reduce onboarding cost, reduce delivery fluctuation, and improve issue localization and collaboration efficiency without changing the team's existing engineering system.

## Quick Start

1. Clone or download this repository to your local workspace.
2. Check the `skills/` directory and select the corresponding Skill by task type.
3. Read the target Skill instructions first, then execute with the current project context.
4. It is recommended to trial-run on small-scope tasks first, and promote it to daily workflows after confirming the output format is consistent with team standards.

> Suggestion: include the Skill usage process in code reviews or task retrospectives to continuously accumulate reusable experience.

## Project Value

- **Standardization**: convert experience-based operations into reusable processes to reduce individual differences.
- **Efficiency improvement**: reduce repetitive work in template setup, migration troubleshooting, optimization analysis, and document maintenance.
- **Collaboration**: unify input/output expectations to facilitate collaboration among development, testing, and documentation roles.
- **Sustainability**: gradually form a team engineering knowledge base through incremental Skill maintenance.
- **Practicality**: emphasize a hands-on orientation of "usable for the current task" rather than abstract methodology.

## Applicable Scenarios

- In the startup stage of new operator/new kernel requirements, basic engineering scaffolds need to be quickly implemented
- Operator performance does not meet expectations, and bottlenecks need structured analysis with optimization suggestions
- Existing CUDA/CPU logic is migrated to Ascend NPU / CANN, and a clear transformation path is needed
- After continuous code evolution, API docs and explanatory docs need synchronized updates and consistency
- The team needs a unified delivery standard to reduce repeated communication and rework

## Repository Structure

```text
skills/
├── ascend-kernel-generator/
├── ascend-kernel-optimization/
├── ascend-npu-migration/
└── ascend-doc-update/
```

## Skills Overview

| Skill | Main Responsibility | Typical Input | Typical Output |
|---|---|---|---|
| `ascend-kernel-generator` | Generate basic scaffolds and templates for kernel / operator development | Operator name, interface constraints, directory conventions | Practical initial code structure and template files |
| `ascend-kernel-optimization` | Assist kernel / operator performance optimization analysis | Performance symptoms, Profiling results, operator implementation | Bottleneck localization conclusions and optimization suggestion list |
| `ascend-npu-migration` | Assist migration to Ascend NPU / CANN platform | Existing CUDA/CPU logic, dependencies, and operator information | Migration steps, transformation key points, and risk notes |
| `ascend-doc-update` | Assist updates and maintenance of technical and explanatory documents | Code changes, interface changes, version information | Synchronized document draft and update item list |

## Skill Descriptions

### 1) ascend-kernel-generator

Used to quickly generate the basic scaffolds and templates needed for Ascend kernel / operator development, suitable for the early requirement stage and rapid onboarding of new members.

- **Input**: Operator/Kernel name, functional target, interface definition, directory and naming conventions
- **Output**: Basic directory structure, templated code scaffolds, follow-up implementation TODO items
- **Problems suitable to solve**: High cost of starting from scratch, inconsistent project structures, time-consuming repeated setup
- **Typical example**: For example, quickly generate a new Ascend kernel development scaffold.

### 2) ascend-kernel-optimization

Used for structured analysis of kernel / operator performance issues, assisting in identifying bottlenecks and defining optimization paths.

- **Input**: Performance indicators, Profiling information, key code paths, existing optimization attempts
- **Output**: Bottleneck judgments, prioritized optimization suggestions, validation focus points
- **Problems suitable to solve**: Slow performance issue localization, unclear optimization direction, high validation cost
- **Typical example**: For example, analyze the performance bottleneck of an operator and provide optimization suggestions.

### 3) ascend-npu-migration

Used to support migration of existing CUDA/CPU-side implementations to Ascend NPU / CANN scenarios and reduce migration risks.

- **Input**: Source implementation logic, dependency capabilities, operator and data flow information, target platform constraints
- **Output**: Migration path suggestions, interface adaptation key points, potential risks and troubleshooting checklist
- **Problems suitable to solve**: High adaptation complexity caused by platform differences, unclear migration boundaries
- **Typical example**: For example, migrate existing CUDA/CPU-side logic to an Ascend NPU scenario.

### 4) ascend-doc-update

Used to synchronously maintain technical documents based on code and interface changes, improving document timeliness and consistency.

- **Input**: Code change points, interface change descriptions, version/release information, documentation conventions
- **Output**: Draft of updated document content, change summary, items pending confirmation
- **Problems suitable to solve**: Outdated documentation, inconsistent interface description and implementation, scattered release information
- **Typical example**: For example, synchronously update interface documentation based on code changes.

## Usage Suggestions

- Clarify the task type first, then select a single Skill as the main workflow entry.
- When linking multiple Skills, it is recommended to proceed in the order of "generation -> optimization/migration -> documentation update".
- Structure input information as much as possible (background, target, constraints, current state) to improve output quality.
- It is recommended that output results enter the review process, rather than directly replacing engineering judgment.
- For high-risk changes, it is recommended to validate on small samples or local modules first before full rollout.

## Typical Examples

- **New operator startup**: Use `ascend-kernel-generator` to generate scaffolds and unify engineering layout and naming.
- **Performance issue troubleshooting**: Use `ascend-kernel-optimization` to perform bottleneck analysis on hotspot paths and form an optimization checklist.
- **Platform migration transformation**: Use `ascend-npu-migration` to sort out transformation boundaries and steps from CUDA/CPU to Ascend.
- **Pre-release documentation finishing**: Use `ascend-doc-update` to synchronize interface changes and complete explanatory docs and change records.

## Target Users

- Ascend / CANN related R&D engineers
- Operator and kernel developers
- Engineering teams responsible for heterogeneous migration and performance optimization
- Project members who need to maintain consistency of technical documentation
- People newly joining the Ascend ecosystem who need to quickly establish engineering methods

## Best Practices

- Accumulate templates by task unit: keep at least one reusable sample for each type of problem.
- Include Skill output in reviews: focus on correctness, maintainability, and boundary conditions.
- Build a "problem-suggestion-validation result" closed loop to continuously calibrate Skill applicability.
- Align with project conventions: keep naming, directories, and document formats consistent.
- Conduct regular retrospectives: update Skill instructions and examples according to actual delivery feedback.

## Contribution Guide

You are welcome to improve repository content incrementally. It is recommended to follow these principles:

- Before adding a new Skill, first clarify the target problem, applicable boundaries, and expected input/output.
- When modifying existing Skills, maintain backward compatibility and add change descriptions.
- Prioritize real, reproducible, and verifiable scenarios for example content.
- Document updates should stay consistent with Skill behavior to avoid mismatch between description and reality.
- It is recommended that submissions focus on a single topic for easier review and traceability.

## Notes

- This repository is positioned as accumulation of R&D assistance capabilities and does not replace final engineering decisions and test verification.
- Skill outputs are usually advisory results and need to be adjusted according to specific project constraints.
- For conclusions involving performance and migration, actual measurement and regression results are recommended as the standard.
- Please avoid including sensitive information and restricted content in documents or examples.
- Before team-wide promotion, it is recommended to complete a small-scope pilot and form usage conventions.

## Disclaimer

The content of this repository is for development reference and assistance use only. Generated results need to be validated with actual engineering scenarios and do not constitute final implementation, performance commitments, or official technical conclusions.

---

