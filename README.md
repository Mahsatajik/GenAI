[Skip to content](https://storm-jacket-4ec.notion.site/Automating-Clinical-Trial-Eligibility-Building-a-Smart-Pipeline-with-LangGraph-ChromaDB-and-Gemin-1dbd34e9912680ed826ccd1d5a991792#main)

# Automating Clinical Trial Eligibility: Building a Smart Pipeline with LangGraph, ChromaDB, and Gemini

April 20, 2025

Clinical trials are the backbone of medical advancements, but identifying eligible patients is a daunting task. Manual screening of patient records against complex trial criteria is time-consuming and error-prone. Our project tackles this challenge with an AI-driven pipeline that automates patient eligibility screening using natural language processing (NLP), vector databases, and large language models (LLMs). In this blog post, we’ll walk you through the workflow, highlight key code components, and share how we overcame technical hurdles to build a robust solution.

### Project Overview

The Clinical Trial Eligibility Pipeline processes clinical discharge notes from the MIMIC-IV dataset, summarizes them, matches them against trial criteria, and evaluates eligibility. Built with Python, LangGraph, ChromaDB, and Google’s Gemini API, the pipeline streamlines the process with a modular design and a user-friendly Gradio interface. Our goal? To accelerate clinical trial recruitment, reduce human error, and make precision medicine more accessible.

Key features:

Summarization: Condenses lengthy clinical notes into concise summaries.

Criteria Processing: Converts trial criteria into actionable LLM prompts.

Chunking & Embedding: Stores summaries in a vector database for efficient retrieval.

Eligibility Evaluation: Matches patient data against criteria using LLMs.

Result Evaluation: Assesses matches with metrics like ROUGE and cosine similarity.

Gradio UI: Visualizes results.

### Workflow Breakdown

The pipeline is orchestrated using LangGraph, a stateful workflow framework. The yellow-dashed boundary in the diagram encapsulates everything controlled by LangGraph, which handles the entire flow of information from input ingestion to final evaluation. Each step is modularized as a LangGraph node, with clean edges controlling the flow from one stage to the next.

Summarize Clinical Data:

Input: Raw clinical discharge notes from MIMIC-IV.

Process: Fine-tunes a Gemini model (

gemini-1.5-flash

) to generate concise summaries, capturing diagnoses, treatments, and outcomes.

Output: A list of summaries stored in

PipelineState\['summaries'\]

.

Database Storage:

Stores summaries and original texts in a SQLite database (

clinical\_data.db

) with unique IDs.

Ensures persistent storage for downstream tasks.

Criteria Processing:

Reads trial criteria from CSV files (e.g., inclusion/exclusion conditions like “CKD with eGFR < 60”).

Generates LLM prompts with chain-of-thought (CoT) steps and lexicons (e.g., “chronic kidney disease”, “eGFR < 60”).

Stores prompts in

PipelineState\['criteria\_prompts'\]

.

Chunking & Embedding:

Splits summaries into chunks (450 characters, 50 overlap) using

RecursiveCharacterTextSplitter

.

Embeds chunks with Google’s

text-embedding-004

model and stores them in a ChromaDB collection (

mimic\_iv\_summary\_chunks\_trial

).

Output: Chunk IDs in

PipelineState\['chunk\_ids'\]

.

RAG :

Queries ChromaDB to retrieve 5 most relevant chunks for each criterion.

Eligibility Evaluation:

Uses Gemini to evaluate eligibility (Yes/No) with evidence from chunks, avoiding inference from symptoms or medications.

Stores results in

PipelineState\['eligibility\_results'\]

.

Result Evaluation:

Uses Reverse RAG and computes metrics (ROUGE, fuzzy matching, cosine similarity) to validate evidence against chunks.

Saves results as CSVs in

/kaggle/working/evaluation\_results/

.

Gradio UI:

Displays existing results (evaluation, database, ChromaDB, logs).

Allows filtering by trial or criteria name.

![](https://storm-jacket-4ec.notion.site/image/attachment%3A96e2ead4-bad1-4037-937f-3a1ddd46f684%3AWorkflow-blogpost.jpg?table=block&id=1dbd34e9-9126-8030-91f0-e7abe3ef1314&spaceId=7a2dddc0-0aad-4b5a-9b6d-f2a7630610e1&width=2000&userId=&cache=v2)

### Key Code Components

Let’s dive into the important functions:

#### 1\. Summarization (  summarize\_text  )

This function fine-tunes a Gemini model to summarize clinical notes. It handles API errors with retries and skips invalid outputs.

defsummarize\_text(state: PipelineState)-> PipelineState:print("Starting summarization step...")
logger.info("Starting summarization step...")
df = pd.read\_csv('/kaggle/input/mimic-iv-dataset/mimic\_iv\_summarization\_test\_dataset\_shortened.csv')
df\_train, df\_test = train\_test\_split(df, test\_size=0.2, random\_state=42)

model\_id ="gemini-1.5-flash-001"\# Fallback modeltry:
tuning\_op = client.tunings.tune(
base\_model="models/gemini-1.5-flash-001-tuning",
training\_dataset={"examples": df\_train\[\['text','summary'\]\].to\_dict(orient='records')},
config=types.CreateTuningJobConfig(batch\_size=16, epoch\_count=2))
model\_id = tuning\_op.name
whilenot client.tunings.get(name=model\_id).has\_succeeded:
time.sleep(60)except Exception as e:
logger.error(f"Failed to fine-tune model: {e}")@retry(wait=wait\_random\_exponential(min=1,max=20), stop=stop\_after\_attempt(15))defgenerate\_summary(text:str)->str:
prompt =f"Summarize the following clinical discharge note in a concise and detailed manner...\\nClinical Note:\\n{text}"
response = client.models.generate\_content(model=model\_id, contents=prompt)
summary = response.candidates\[0\].content.parts\[0\].text
return summary if summary.strip()andlen(summary)>=50else"(error: invalid summary)"

state\['summaries'\]=\[generate\_summary(text)for text in state\['raw\_text'\]if text.strip()\]
logger.info(f"Generated {len(state\['summaries'\])} valid summaries.")return state

​

#### 2\. Chunking & Embedding (  chunk\_and\_embed  )

This function splits summaries into chunks and stores them in ChromaDB.

defchunk\_and\_embed(state: PipelineState)-> PipelineState:
chroma\_client = chromadb.PersistentClient(path="/kaggle/working/chromadb")classGeminiEmbeddingFunction(EmbeddingFunction):@retry(wait=wait\_random\_exponential(min=1,max=10), stop=stop\_after\_attempt(5))def\_\_call\_\_(self,input: Documents)-> Embeddings:
response = client.models.embed\_content(
model="models/text-embedding-004",
contents=input,
config=types.EmbedContentConfig(task\_type="retrieval\_document"))
embeddings =\[e.values for e in response.embeddings\]
logger.info(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings\[0\])}")return embeddings

embed\_fn = GeminiEmbeddingFunction()
state\['embedding\_function'\]= embed\_fn

splitter = RecursiveCharacterTextSplitter(chunk\_size=450, chunk\_overlap=50)
all\_chunks =\[\]
chunk\_ids =\[\]for idx, summary inenumerate(state\['summaries'\]):if"(error"in summary.lower():continue
chunks = splitter.split\_text(summary)
all\_chunks.extend(chunks)
chunk\_ids.extend(\[f"{state\['summary\_ids'\]\[idx\]}\_{i}"for i inrange(len(chunks))\])try:
chroma\_client.delete\_collection(name="mimic\_iv\_summary\_chunks\_trial")except:pass

collection = chroma\_client.create\_collection(
name="mimic\_iv\_summary\_chunks\_trial",
embedding\_function=embed\_fn,
metadata={"hnsw:space":"cosine"})if all\_chunks:
collection.add(documents=all\_chunks, ids=chunk\_ids)
logger.info(f"Stored {collection.count()} chunks in ChromaDB.")

state\['chunk\_ids'\]= chunk\_ids
return state

​

#### 3\. Criteria Processing

Trial criteria from CSV files (e.g., /kaggle/input/clinical-trial-sample) are converted into JSON prompts for LLMs.

CSV to JSON:

Reads CSVs with Criteria\_name and Content columns.

Saves as JSON (e.g., trial\_name.json).

Prompt Generation:

Creates chain-of-thought (CoT) prompts emphasizing exact matches in clinical notes.

Limits prompts to 2 sentences and <10 CoT steps.

defprocess\_criteria(state: PipelineState)-> PipelineState:
logger.info("Starting criteria processing step...")
csv\_folder ="/kaggle/input/clinical-trial-sample"
csv\_files = glob.glob(os.path.join(csv\_folder,"\*.csv"))
criteria\_prompts =\[\]for file\_path in csv\_files:
df = pd.read\_csv(file\_path)\[\['Criteria\_name','Content'\]\]
base\_name = os.path.splitext(os.path.basename(file\_path))\[0\]
json\_filename =f"/kaggle/working/{base\_name}.json"
df.to\_json(json\_filename, orient='records')withopen(json\_filename,"r")as f:
data = json.load(f)for item in data:
criteria\_prompts.append({"Trial\_Name": base\_name,"Criteria\_name": item\['Criteria\_name'\],"Text": item\['Content'\]\[:200\],\# Simplified for brevity"chainOfThought":\["Step 1: Identify exact match","Step 2: Validate context"\],"extractableLexicons":\["diagnosis","procedure"\]})
logger.info(f"Converted {file\_path} to {json\_filename}.")
state\['criteria\_prompts'\]= criteria\_prompts
state\['trial\_names'\]=\[os.path.splitext(os.path.basename(f))\[0\]for f in csv\_files\]
logger.info(f"✅ Processed {len(criteria\_prompts)} criteria prompts.")return state

​

#### 4\. Eligibility Evaluation (  evaluate\_eligibility  )

This function queries ChromaDB for relevant chunks and uses Gemini to evaluate eligibility.

defevaluate\_eligibility(state: PipelineState)-> PipelineState:
chroma\_client = chromadb.PersistentClient(path="/kaggle/working/chromadb")
collection = chroma\_client.get\_collection(name="mimic\_iv\_summary\_chunks\_trial", embedding\_function=state\['embedding\_function'\])

eligibility\_results =\[\]ifnot state\['chunk\_ids'\]ornot state\['summaries'\]:for sid in state\['summary\_ids'\]:for criterion in state\['criteria\_prompts'\]:
eligibility\_results.append({"summary\_id": sid,"criteria\_name": criterion\['Criteria\_name'\],"trial\_name": criterion.get('Trial\_Name','Unknown'),"eligibility":"No","evidence":"No chunks available"})
state\['eligibility\_results'\]= eligibility\_results
return state

for criterion in state\['criteria\_prompts'\]:
criteria\_name = criterion\['Criteria\_name'\]
criteria\_text = criterion\['Text'\]
cot\_steps = criterion\['chainOfThought'\]
lexicons = criterion\['extractableLexicons'\]
trial\_name = criterion.get('Trial\_Name','Unknown')

results = collection.query(query\_texts=\[criteria\_text.replace("\\n"," ")\], n\_results=5)
relevant\_chunks = results\['documents'\]\[0\]
chunk\_ids = results\['ids'\]\[0\]
context ="\\n".join(\[f"Chunk {i+1} (ID: {cid}): {chunk}"for i,(chunk, cid)inenumerate(zip(relevant\_chunks, chunk\_ids))\])or"No relevant chunks found."

prompt =f"""
You are a medical data analyst evaluating patient eligibility for a clinical trial based on a specific criterion.
Review the provided patient summary chunks and determine if the patient meets the criterion.
Follow these instructions:
\- Do not infer conditions based on symptoms or medications.
\- Only consider conditions explicitly mentioned in the chunks, pertinent to the patient (not their family).
\- Use the provided chain-of-thought steps to guide your reasoning.
\- If no evidence is found, conclude the patient is not eligible.
\- Respond with a concise explanation of eligibility (Yes/No) and any supporting evidence from the chunks.
\- Include the summary ID in your response.

Criterion: {criteria\_text}
Extractable Lexicons: {', '.join(lexicons)if lexicons else'None'}
Chain of Thought:
{chr(10).join(\[f"- {step}"for step in cot\_steps\])}

Patient Summary Chunks:
{context}

Summary IDs to Evaluate:
{', '.join(state\['summary\_ids'\])}

Response Format:
For each summary ID, provide a brief statement like:
"Summary ID: <ID>, Eligibility: Yes/No, Evidence: <evidence or 'None'>"
Separate responses for multiple IDs with newlines.
"""@retry(wait=wait\_random\_exponential(min=5,max=60), stop=stop\_after\_attempt(10))defgenerate\_eligibility():
response = client.models.generate\_content(model="gemini-1.5-flash", contents=prompt)
response\_text = response.candidates\[0\].content.parts\[0\].text.strip()ifnot response\_text:raise ValueError("Empty response from API")return response\_text

try:
response\_text = generate\_eligibility()
lines = response\_text.split('\\n')
result\_map ={}for line in lines:if'Summary ID'notin line:continue
parts = line.split(', ')iflen(parts)<3:continue
sid = parts\[0\].split(': ')\[1\]
eligibility = parts\[1\].split(': ')\[1\]
evidence = parts\[2\].split(': ')\[1\]
result\_map\[sid\]={"Eligibility": eligibility,"Evidence": evidence if evidence !='None'elseNone}for sid in state\['summary\_ids'\]:if sid in result\_map:
eligibility\_results.append({"summary\_id": sid,"criteria\_name": criteria\_name,"trial\_name": trial\_name,"eligibility": result\_map\[sid\]\["Eligibility"\],"evidence": result\_map\[sid\]\["Evidence"\]})else:
eligibility\_results.append({"summary\_id": sid,"criteria\_name": criteria\_name,"trial\_name": trial\_name,"eligibility":"No","evidence":"No response from API"})
time.sleep(15)\# Avoid quota limitsexcept Exception as e:for sid in state\['summary\_ids'\]:
eligibility\_results.append({"summary\_id": sid,"criteria\_name": criteria\_name,"trial\_name": trial\_name,"eligibility":"No","evidence":f"Error: {str(e)}"})

state\['eligibility\_results'\]= eligibility\_results
logger.info(f"Generated {len(eligibility\_results)} eligibility results.")return state

​

#### 5\. Eligibility Evaluation

Patient eligibility is assessed by querying ChromaDB and using LLMs to match criteria.

RAG Query:

Queries ChromaDB for top-5 relevant chunks per criterion.

LLM Evaluation:

Uses Gemini 1.5 Flash with CoT prompts to determine eligibility.

Ensures exact matches (no inference from symptoms/medications).

defevaluate\_eligibility(state: PipelineState)-> PipelineState:
logger.info("Starting eligibility checking step...")
chroma\_client = chromadb.PersistentClient(path=CHROMA\_DB\_PATH)
collection = chroma\_client.get\_collection(name="mimic\_iv\_summary\_chunks\_trial", embedding\_function=state\['embedding\_function'\])
eligibility\_results =\[\]for criterion in state\['criteria\_prompts'\]:
results = collection.query(query\_texts=\[criterion\['Text'\].replace("\\n"," ")\], n\_results=5)
context ="\\n".join(\[f"Chunk {i+1}: {chunk}"for i, chunk inenumerate(results\['documents'\]\[0\])\])
prompt =f"Evaluate patient eligibility...\\nCriterion: {criterion\['Criteria\_name'\]}\\nRetrieved Chunks: {context}"
response = client.models.generate\_content(model="gemini-1.5-flash", contents=prompt)
results = json.loads(response.candidates\[0\].content.parts\[0\].text.replace("\`\`\`json","").replace("\`\`\`",""))for sid in state\['summary\_ids'\]:
result =next((r for r in results if r\["summary\_id"\]== sid),None)
eligibility\_results.append({"summary\_id": sid,"criteria\_name": criterion\['Criteria\_name'\],"trial\_name": criterion.get('Trial\_Name','Unknown'),"eligibility": result\["Eligibility"\]if result else"No","evidence": result\["Evidence"\]if result else"Missing API response"})
state\['eligibility\_results'\]= eligibility\_results
logger.info(f"✅ Generated {len(eligibility\_results)} eligibility results.")return state

​

#### 6\. Gradio UI

The Gradio interface lets users view results, addressing the need to explore existing outputs.

import gradio as gr
import pandas as pd
import sqlite3
import chromadb

defload\_evaluation\_results(trial\_filter="", criteria\_filter=""):
csv\_path ="/kaggle/working/evaluation\_results/all\_results.csv"ifnot os.path.exists(csv\_path):return pd.DataFrame(),"No evaluation results found."
df = pd.read\_csv(csv\_path)if trial\_filter:
df = df\[df\['Trial\_Name'\].str.contains(trial\_filter,case=False, na=False)\]if criteria\_filter:
df = df\[df\['Criteria\_Name'\].str.contains(criteria\_filter,case=False, na=False)\]return df,f"Loaded {len(df)} evaluation results."defload\_database\_contents():
db\_path ="/kaggle/working/clinical\_data.db"ifnot os.path.exists(db\_path):return pd.DataFrame(),"No database found."with sqlite3.connect(db\_path)as conn:
df = pd.read\_sql\_query("SELECT id, original\_text, summary FROM summaries", conn)
df\["Original\_Text"\]= df\["original\_text"\].apply(lambda x: x\[:100\]+"...")
df\["Summary"\]= df\["summary"\].apply(lambda x: x\[:100\]+"...")return df,f"Loaded {len(df)} database records."defload\_chromadb\_contents():
chroma\_client = chromadb.PersistentClient(path="/kaggle/working/chromadb")
collection = chroma\_client.get\_collection(name="mimic\_iv\_summary\_chunks\_trial")
count = collection.count()
results = collection.peek(limit=5)
data =\[{"Chunk\_ID": cid,"Document": doc\[:100\]+"..."}for cid, doc inzip(results\['ids'\], results\['documents'\])\]
data.insert(0,{"Chunk\_ID":"Total Count","Document":str(count)})return pd.DataFrame(data),f"Loaded {count} ChromaDB chunks."defload\_logs():withopen("/kaggle/working/pipeline.log",'r')as f:return f.read()or"Log file is empty."with gr.Blocks()as demo:
gr.Markdown("# Clinical Trial Pipeline Results Viewer")with gr.Row():with gr.Column():
trial\_filter = gr.Textbox(label="Filter by Trial Name")
criteria\_filter = gr.Textbox(label="Filter by Criteria Name")
clear\_button = gr.Button("Clear Outputs")with gr.Column():
status = gr.Textbox(label="Status", value="Ready to view results.")with gr.Tabs():with gr.Tab("Evaluation Results"):
evaluation\_output = gr.DataFrame(label="Evaluation Results")with gr.Tab("Database Contents"):
db\_output = gr.DataFrame(label="Database Contents")with gr.Tab("ChromaDB Contents"):
chroma\_output = gr.DataFrame(label="ChromaDB Contents")with gr.Tab("Logs"):
log\_output = gr.Textbox(label="Logs", lines=20)

consolidated\_csv = gr.File(label="Download Consolidated Results", value="/kaggle/working/evaluation\_results/all\_results.csv")defupdate\_outputs(trial\_filter, criteria\_filter):
eval\_df, eval\_status = load\_evaluation\_results(trial\_filter, criteria\_filter)
db\_df, db\_status = load\_database\_contents()
chroma\_df, chroma\_status = load\_chromadb\_contents()
logs = load\_logs()return{"status":f"{eval\_status}\\n{db\_status}\\n{chroma\_status}","evaluation\_output": eval\_df,"db\_output": db\_df,"chroma\_output": chroma\_df,"log\_output": logs,"consolidated\_csv":"/kaggle/working/evaluation\_results/all\_results.csv"}

trial\_filter.change(fn=update\_outputs, inputs=\[trial\_filter, criteria\_filter\], outputs=\[status, evaluation\_output, db\_output, chroma\_output, log\_output, consolidated\_csv\])
criteria\_filter.change(fn=update\_outputs, inputs=\[trial\_filter, criteria\_filter\], outputs=\[status, evaluation\_output, db\_output, chroma\_output, log\_output, consolidated\_csv\])
clear\_button.click(fn=clear\_outputs, inputs=None, outputs=status)

demo.launch(server\_name="0.0.0.0", server\_port=7860, inline=True)

​

Feature Highlight: The UI let users filter and view results (e.g., eligibility for CKD trials) interactively.

### Impact & Future Work

This pipeline demonstrates how AI can transform clinical trial recruitment, saving time and improving accuracy. By automating eligibility screening, we pave the way for faster trials and better patient outcomes.

Future enhancements:

Add real-time visualization of eligibility metrics.

Expand to multi-modal data (e.g., imaging reports).

### Get Involved

Try the pipeline and explore the code on Kaggle, or share your feedback on Notion! Let’s make clinical trials more efficient together.

Built with ![❤️](<Base64-Image-Removed>) by [Mehrnoosh Azizi](https://www.linkedin.com/in/mehrnoosh-azizi/), [Reza Rezaeian](https://www.linkedin.com/in/reza-rezaeian/) and [Mahsa Tajik](https://www.linkedin.com/in/mahsa-tajik/).
