# **Construction Arbitration Multi-Agent Pipeline**

## **1. Objective**

The objective is to design and implement a multi-phase LLM-driven pipeline to analyze ~50 selected documents from a single construction arbitration matter.


The system will:

- Convert OCR text into structured representations
    
- Organize documents into meaningful categories
    
- Discover financial claim heads from the corpus
    
- Map evidence to claims with reasoning
    
- Generate a structured case-level analytical report
    

  

The pipeline is designed to be modular, resumable, parallelizable, and grounded in document evidence.



## **2. Assumptions**

- All selected documents belong to a single arbitration.
    
- OCR text quality may vary.
    
- Claim heads must be discovered from corpus evidence, not predefined.
    
- Completeness of claim head coverage is slightly prioritized over extreme specificity.
    
- A single document may relate to multiple claim heads and may support different parties on different issues.
    
- The system must avoid hallucinating unsupported claims.
    



## **3. Corpus Selection Strategy**


Documents were selected to ensure lifecycle and dispute diversity:

  

### **Contract Formation**

- Contract Agreement
    
- Work Order
    
- Supplementary Work Order
    

  

### **Execution Phase**

- Milestone schedule (baseline programme)
    
- RA Bill submission
    
- Extra item submissions
    
- Escalation bill submissions
    

  

### **Delay / EOT**

- Extension of time application
    
- Delay appeal and hindrance letters
    
- COVID impact notice
    
- Force majeure memorandum
    

  

### **Financial Disputes**

- Escalation invoices
    
- Final bill certificate
    
- Final bill dispute correspondence
    
- Mobilization advance interest deduction
    

  

### **Liquidated Damages / Compensation**

- Government levy orders
    
- Compensation orders
    

  

### **Security / Bank Guarantee**

- BG return request
    
- Performance security reduction
    
- BG amendment
    

  

### **Legal Escalation**

- Claim statement
    
- Demand notice
    
- Legal notice reply
    
- Court filing
    
- Writ petition
    
- Arbitration board resolution
    

  

This ensures coverage of:

- Delay claims
    
- Prolongation claims
    
- Escalation claims
    
- Extra item/deviation claims
    
- Final bill/payment claims
    
- Liquidated damages counterclaims
    
- Bank guarantee disputes
    
- Arbitration escalation
    



## **4. Pipeline Overview**

```markdown
segments/
   ↓
Phase 1: Structured Index Extraction
   ↓
index.json
   ├── Phase 2: Document Type Taxonomy & Classification
   └── Phase 3: Claim Head Discovery & Mapping
            ↓
Phase 4: Case-Level Reasoning & Narrative
```




Phase 2 and Phase 3 operate in parallel after Phase 1.

Phase 4 depends on all prior outputs.

# **Phase 1 – Index & Structured Extraction**

  

### **Goal**

  

Convert OCR text into structured metadata.

  

### **Input**

- Raw OCR text per segment
    

  

### **Output**

  

`index.json`

  

### **Per-Segment Schema**
```json
{
  "segment_id": "...",
  "document_type": "...",
  "document_stage": "...",
  "parties": [...],
  "dates": [...],
  "monetary_amounts": [...],
  "summary": "...",
  "dispute_signals": [...]
}
```
### Design Decisions

- Strict JSON schema enforcement
- Automatic retry on malformed outputs
- Explicit instruction to ignore OCR artifacts
- Monetary values normalized to structured numeric format
- Dates normalized to consistent format
- Dispute signals extracted as structured flags (e.g., delay, escalation, BG, LD, payment dispute)


### **Parallelization**  

Each segment processed independently → fully parallelizable.




# **Phase 2 – Document Type Organization**

## **Step 1: Taxonomy Generation (Corpus-Level)**


Input:

- `index.json`
    

  

Output:

- `type_taxonomy.json`
    

  

The taxonomy will balance:

- Legal interpretability
    
- Analytical usefulness for downstream reasoning
    

  

Expected top-level clusters:

- Contractual Framework
    
- Execution & Progress
    
- Delay / EOT
    
- Financial & Billing
    
- Security & Guarantees
    
- Legal Escalation
    
- Government / Regulatory Orders
    



## **Step 2: Per-Segment Classification**

  

Input:

- `index.json`
    
- `type_taxonomy.json`
    

  

Output:

- `type_classifications.json`
    

  

Each segment:

- Assigned 1 primary cluster
    
- Optional secondary cluster
    
- Confidence score
    
- Reasoning explanation
    

  

No new categories introduced during classification.



# **Phase 3 – Claim Head Discovery & Mapping**

  

## **Step 1: Claim Head Discovery (Corpus-Level)**

  

Input:

- `index.json`
    

  

Output:

- `claim_heads.json`
    

  

### **Strategy**

- Analyze summaries, monetary references, dispute signals.
    
- Identify distinct financial claim categories.
    
- Balance completeness and specificity.
    
- Merge closely related themes.
    
- Avoid fragmenting into micro-heads.
    
### Claim Head Qualification Criteria

A claim head must:
- Represent a financially distinct dispute theme
- Be supported by at least two documentary references
- Not be purely procedural unless tied to monetary impact
- Not be subdivided unless financial or legal treatment materially differs

  
### Structural Boundary

Step 1 establishes the structural claim hierarchy and high-level documentary anchors.
Detailed segment-to-claim mapping occurs in Step 2.

  

### Claim Head Schema

```json
{
  "claim_key": "...",
  "title": "...",
  "description": "...",
  "claimant": "...",
  "approx_total_amount": "...",
  "sub_heads": [
    {
      "sub_claim_key": "...",
      "title": "...",
      "description": "...",
      "approx_amount": "...",
      "supporting_segment_ids": [...]
    }
  ],
  "supporting_segment_ids": [...]
}
```
- `approx_total_amount` is derived from aggregation of sub-head components where available.
- `supporting_segment_ids` enforce corpus-grounded discovery.

## **Step 2: Claim Mapping (Per-Segment)**

  

Input:

- `index.json`
    
- `claim_heads.json`
    

  

Output:

- `claim_classifications.json`
    

  

Each segment may map to multiple claims.

  

For each mapping:

```json
{
  "segment_id": "...",
  "claim_key": "...",
  "relevance_type": "direct | contextual | rebuttal",
  "party_role": "supports_contractor | supports_employer | neutral",
  "confidence": "...",
  "reasoning": "..."
}
```

Confidence is computed independently per claim mapping.


# **Phase 4 – Case-Level Reasoning & Report**

  

Input:

- `index.json`
    
- `type_classifications.json`
    
- `claim_heads.json`
    
- `claim_classifications.json`
    

  

Output:

- `case_analysis.md`
    

  

### **Sections**

- Case Overview
    
- Document Landscape
    
- Timeline of Key Events
    
- Claim Head Analysis
    
- Evidence Coverage Matrix
    
- Observations & Gaps
    
- Potential Risk Areas
    

  

Narrative grounded strictly in structured outputs.

  

No unsupported extrapolation.


## **5. State Management & Checkpointing**

  

Each phase:

- Writes deterministic output file
    
- Checks if output exists before execution
    
- Skips execution if already completed
    
- Logs execution status clearly
    

  

Pipeline is resumable at any phase.


## **6. Model Selection Strategy**

- Phase 1: Cost-efficient structured extraction model
    
- Phase 2: Mid-tier reasoning model
    
- Phase 3: Strong reasoning model for claim discovery
    
- Phase 4: Highest reasoning model for synthesis
    

  

Model selection is aligned with reasoning complexity and cost efficiency.


## **7. Hallucination Control**

- Claim heads must reference supporting documents.
    
- No claim head without documentary anchors.
    
- All monetary values must be extracted from index.json.
    
- Cross-check reasoning against source segments.
    



## **8. Scalability Considerations**

  

For larger corpora:

- Phase 1 and claim mapping fully parallelizable.
    
- Claim discovery chunkable with summarization layers.
    
- Embedding-based clustering can pre-group documents.
    
- Incremental re-indexing supported.
    

## 9. Risks & Mitigations

| Risk                        | Mitigation                  |
| --------------------------- | --------------------------- |
| OCR noise                   | Strict parsing + retry      |
| Over-fragmented claim heads | Merge related themes        |
| Missed claim categories     | Prioritize completeness     |
| Hallucination               | Require document references |
| Context window overflow     | Use summarized index        |


---

# Selected Working Set (50 Segments) & Rationale

## Selected Segments

The following ~50 segments were selected from the full corpus to ensure representative coverage across the contract lifecycle and dispute escalation stages:

- Abstract_Statement_-_Construction_Work_Deviation
- Annexure_-_Construction_Milestones_ISITE_Bangalore
- Application_-_Extension_of_Time_for_Construction
- Board_Resolution_-_Arbitration_Authorization
- Certificate_-_Final_Bill_for_Construction_Work
- Claim_Abstract_-_Covid_and_Prolongation_Charges
- Claim_Letter_-_Construction_Dispute_Settlement
- Claim_Statement_-_Construction_Project_AITF-1
- Clarification_Letter_-_EOT_Crane_Work
- Completion_Certificate_-_AITF-1_Extension_Work
- Contract_Agreement_-_Construction_of_AITF-1_Facility
- Court_Filing_-_Construction_Dispute
- Demand_Notice_-_Construction_Project_Delays
- Financial_Report_-_Lockdown_Expenses
- Government_Order_-_Compensation_for_Construction_Delay
- Government_Order_-_Levy_of_Compensation
- Government_Order_-_Performance_Bank_Guarantee_Reduction
- Invoice_-_Construction_Escalation_Bill
- Invoice_-_Construction_Escalation_Bill_2
- Legal_Notice_-_Demand_Notice_Reply
- Letter_-_BG_Return_for_Construction_Work
- Letter_-_Bank_Guarantee_Extension_for_ISRO_Project
- Letter_-_Clarifications_on_EOT_Crane_Design
- Letter_-_Construction_Delay_Appeal
- Letter_-_Construction_Delay_Hindrances
- Letter_-_Escalation_Bill_Submission
- Letter_-_Extension_of_Time_Dispute
- Letter_-_Extension_of_Time_Request
- Letter_-_Extra_Item_Claim_Rejection
- Letter_-_Final_Bill_Certification_Dispute
- Letter_-_Final_Bill_Payment_Request
- Letter_-_Mobilisation_Advance_Interest_Deduction
- Letter_-_Payment_Release_Request
- Letter_-_Performance_Bank_Guarantee_Amendment
- Letter_-_Rate_Analysis_for_EOT_Crane_Work
- Letter_-_Request_for_BG_Release
- Letter_-_Submission_of_Extra_Item_2
- Letter_-_Submission_of_Extra_Item_2_2
- Letter_-_Submission_of_RA_Bill_No_07
- Letter_-_Waiver_Request_for_Interest_on_Advance
- Non-Starter_Report_-_Pre-Institution_Mediation
- Notice_-_COVID-19_Impact_on_Contract
- Office_Memorandum_-_Force_Majeure_Clause
- Office_Memorandum_-_Performance_Security_Reduction
- Petition_-_Pending_Contractor_Payment
- Reply_Letter_-_Construction_Extension_Dispute
- Statement_of_Truth_-_Commercial_Suit
- Supplementary_Work_Order_-_AITF-1_Construction
- Work_Order_-_AITF_Extension_Construction
- Writ_Petition_-_Payment_Dispute

## Selection Rationale

The selected segments were curated to ensure lifecycle coverage and claim-head completeness, rather than alphabetical or random selection.

1. Contractual Framework
    - Contract Agreement
    - Work Order
    - Supplementary Work Order

These documents establish contractual scope, obligations, variation framework, and risk allocation. They are essential for interpreting delay, escalation, and compensation claims.

2. Execution & Performance Evidence
    - Milestone Annexure
    - RA Bill submission
    - Completion Certificate
    - Extra item submissions
    - Escalation bill invoices

These documents provide performance baseline, billing events, and cost variation evidence. They are foundational to unpaid bill and escalation claim heads.

3. Delay & Prolongation
    - Extension of Time applications
    - Hindrance letters
    - Delay appeals
    - COVID impact notice
    - Force majeure memorandum
    - Lockdown expense report

These segments are central to delay-based claim heads including prolongation costs, overhead recovery, and time extensions.

4. Financial & Payment Disputes
    - Final Bill Certificate
    - Final Bill Dispute correspondence
    - Payment release requests
    - Mobilization advance interest deduction
    - Escalation invoices

These documents directly inform unpaid amount, escalation, and interest-related claims.

5. Liquidated Damages & Compensation
    - Government levy orders
    - Compensation orders

These documents potentially form employer counterclaims or liquidated damages deductions.

6. Security & Bank Guarantee Issues
    - BG return request
    - BG extension
    - Performance security reduction
    - BG amendment

These segments enable discovery of bank guarantee and security-related claim heads.


7. Arbitration & Legal Escalation
    - Claim Statement
    - Claim Abstract
    - Demand Notice
    - Legal Notice Reply
    - Court Filing
    - Writ Petition
    - Arbitration Board Resolution
    - Statement of Truth

These documents crystallize the dispute and define the formal claim structure, making them critical inputs for claim head discovery.


### Why This Subset is Sufficient

This subset:
    - Covers the full contract → execution → dispute → arbitration lifecycle
    - Contains both claimant and respondent perspectives
    - Includes monetary instruments (invoices, abstracts, certificates)
    - Includes rebuttals and counterclaims
    - Enables discovery of all major claim heads without overloading the working set

The selection prioritizes diversity of dispute themes over volume, ensuring claim head completeness while remaining computationally manageable (~50 segments).


# **Final Note**

  
The system is designed not as a simple classification tool, but as a structured dispute intelligence engine capable of:

- Extracting structured evidence
    
- Organizing documents
    
- Discovering financial dispute structure
    
- Generating coherent arbitration analysis