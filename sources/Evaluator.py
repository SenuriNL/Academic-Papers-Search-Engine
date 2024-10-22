from file_operations import retrieve_data
from text_processing import preprocess_paper
from sklearn.metrics import precision_score, recall_score, f1_score
from SearchEngine import SearchEngine
import matplotlib.pyplot as plt
class Evaluator:
    def __init__(self, search_engine, ground_truth):
        self.search_engine = search_engine
        self.ground_truth = ground_truth

    def evaluate(self, queries):
        results = {
            'boolean': [],
            'vsm': [],
            'okapi_bm25': [],
        }
        for query in queries:
            boolean_papers, vsm_papers = self.search_engine.search(query)

            # Evaluate Boolean retrieval
            boolean_metrics = self.calculate_metrics(boolean_papers)
            results['boolean'].append(boolean_metrics)

            # Evaluate VSM retrieval
            vsm_metrics = self.calculate_metrics(vsm_papers)
            results['vsm'].append(vsm_metrics)

        return results

    def calculate_metrics(self, retrieved_docs):
        y_true = [1 if doc_id in self.ground_truth else 0 for doc_id in retrieved_docs]
        y_pred = [1] * len(retrieved_docs)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    def plot_metrics(self,algorithm, results, queries):
        for metric_name in ['precision', 'recall', 'f1']:
            metric_values = [metrics[metric_name] for metrics in results[algorithm]]
            plt.figure(figsize=(10, 5))
            plt.bar(queries, metric_values, color='blue', alpha=0.7)
            plt.title(f"{algorithm} {metric_name.capitalize()} Scores")
            plt.xlabel('Queries')
            plt.ylabel(metric_name.capitalize())
            plt.tight_layout()
            plt.show()


papers_collection = retrieve_data('C:\\Users\\rmct2\\OneDrive - Sri Lanka Institute of Information Technology\\Desktop\\SLIIT\\Y3S1\\IRWA\\IRWA\\Academic-Papers-Search-Engine\\datasets\\arXiv_papers_less.json')
preprocessed_metadata = {}
for paper in papers_collection:
    document_id = paper['arXiv ID']
    preprocessed_metadata[document_id] = preprocess_paper(paper)

search_engine = SearchEngine(preprocessed_metadata)
search_engine.build_inverted_index()
queries = [
    "Investigating the Impact of Climate Change on Global Ecosystems and Biodiversity Conservation Strategies", # Long Query
    "Renewable Energy Sources", # Short Query
    "Analyzing the Interplay of Quantum Algorithms and Machine Learning Models in Real-world Applications", # Complex Query
    "Process", # Simple Query
    "Hanyu Li, Wenhan Huang", # Authors Query
    "Information Retrieval", # Subjects Query
    "cs.AI", # Subject Tags Query
    "Recent Developments in Dark Matter Detection Technologies", # Title Query
    "July 2023" # Date Query
]
total_ground_truth = []
for query in queries:
    _,_,ground_truth = search_engine.search(query)
    total_ground_truth += ground_truth
evaluator = Evaluator(search_engine, total_ground_truth)
results = evaluator.evaluate(queries)
for algorithm, metrics_list in results.items():
    evaluator.plot_metrics(algorithm, results, queries)
    
for algorithm, metrics_list in results.items():
    print(f"\nMetrics for {algorithm} retrieval:")
    for i, metrics in enumerate(metrics_list, start=1):
        print(f"{queries[i-1]}: Precision={metrics['precision']}, Recall={metrics['recall']}, F1={metrics['f1']}")
