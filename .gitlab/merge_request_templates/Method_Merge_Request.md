<!-- .gitlab/merge_requests_templates/Method_Merge_Request.md -->

## 🔀 Method Change Merge Request

### TL;DR
Briefly describe the method change (max 5 sentences).

### Method Details
**Method Name**: `method_name`

**Time Series Types**:
- [ ] 🌐 All
- [ ] 🌱 Biomass
- [ ] ❄️ Cooling (tick only for methods that exclusively generate cooling, otherwise choose HVAC)
- [ ] 🔆 CSP (Concentrated Solar Power)
- [ ] 🚿 DHW (Domestic Hot Water)
- [ ] ⚡ Electricity
- [ ] 🌋 Geothermal
- [ ] 🔥 Heating (tick only for methods that exclusively generate cooling, otherwise choose HVAC)
- [ ] 🏢 HVAC (Heating, Ventilation, and Air Conditioning)
- [ ] 💧 Hydro
- [ ] 🚗 Mobility
- [ ] 👥 Occupancy
- [ ] ☀️ PV (Photovoltaic)
- [ ] 〰️ Tidal
- [ ] 🌊 Wave
- [ ] 💨 Wind
- [ ] 🔍 Other: _____________

**Type of change**:
- [ ] ✨ New method
- [ ] 🔄 Method modification
- [ ] 🐛 Bug fix in method
- [ ] ♻️ Method refactoring
- [ ] ⚡ Method performance improvement
- [ ] 📝 Method documentation update

**Affected Components/Modules**: 
- List the components or modules affected by this change (if applicable)

**⚠️ Breaking Change**: 
- [ ] Yes: (If yes, briefly explain which existing functionality might be affected)
- [ ] No

### Linked Issues
Link any relevant issues by writing `Closes #issue_number`.

### Implementation Details
- **Input Parameters**: List and describe the parameters
- **Return Value**: Describe what the method returns
- **Dependencies**: List any other methods/modules this method depends on
- **Algorithm/Logic**: Brief explanation of how the method works

### Changes Made
For new methods: Explain why this method was needed and how it fits into the existing architecture.
For modifications: Clearly describe what was changed and why.
For fixes: Describe the bug and how your changes fix it.

### Alternatives Considered
Briefly describe alternative approaches that were considered and why the implemented solution was chosen.

### How to Test
Provide specific test cases for this method:
1. Test case 1: Input → Expected output
2. Test case 2: Input → Expected output
3. Edge cases tested:
   - Edge case 1
   - Edge case 2

### Documentation
- [ ] Method has proper docstrings
- [ ] Documentation has been updated (if applicable)
- [ ] Examples of usage have been provided

### Performance Considerations
Discuss any performance implications of this method (time complexity, memory usage, etc.)
- **Time / 100 profiles (single-core | quad-core)**: e.g. 5 | 15 s
- **Memory**: (if applicable/relevant)
- **Time Complexity**: e.g. log(O) (if applicable)

### Checklist
- [ ] My method follows the project's naming conventions and style guidelines
- [ ] I have reviewed my own code for clarity and maintainability
- [ ] I have added appropriate error handling
- [ ] I have added appropriate logging (if applicable)
- [ ] I have added unit tests for this method
- [ ] I have updated the documentation as needed.
- [ ] All tests pass locally with my changes
- [ ] I have considered edge cases and potential failure modes

### Additional Notes
Add any additional notes about the method implementation, design decisions, or areas where reviewer feedback would be particularly helpful.

### How to Make a Good PR
Watch this [How to make a good PR (Video)](https://www.youtube.com/watch?v=_HedItVFr5M).
