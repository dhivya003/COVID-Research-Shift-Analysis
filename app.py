from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import networkx as nx
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from scipy.spatial.distance import cosine
from functools import lru_cache
import os
import json
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# List of domains
domains = [
    'AIDS & HIV', 'Addiction', 'Alternative & Traditional Medicine', 'Anesthesiology',
    'Audiology, Speech & Language Pathology', 'Bioethics', 'Biomedical Technology', 'Cardiology',
    'Child & Adolescent Psychology', 'Clinical Laboratory Science', 'Communicable Diseases',
    'Critical Care', 'Dentistry', 'Dermatology', 'Developmental Disabilities', 'Diabetes',
    'Emergency Medicine', 'Endocrinology', 'Epidemiology', 'Gastroenterology & Hepatology',
    'Genetics & Genomics', 'Gerontology & Geriatric Medicine', 'Gynecology & Obstetrics',
    'Heart & Thoracic Surgery', 'Hematology', 'Hospice & Palliative Care', 'Immunology',
    'Medical Informatics', 'Medical Chemistry', 'Molecular Biology', 'Natural Medicines & Medicinal Plants',
    'Neurology', 'Neurosurgery', 'Nuclear Medicine, Radiotherapy & Molecular Imaging', 'Nursing',
    'Nutrition Science', 'Obesity', 'Oncology', 'Ophthalmology & Optometry', 'Oral & Maxillofacial Surgery',
    'Orthopedic Medicine & Surgery', 'Otolaryngology', 'Pain & Pain Management', 'Pathology',
    'Pediatric Medicine', 'Pharmacology & Pharmacy', 'Physical Education & Sports Medicine', 'Physiology',
    'Plastic & Reconstructive Surgery', 'Pregnancy & Childbirth', 'Primary Health Care', 'Psychiatry',
    'Psychology', 'Public Health', 'Pulmonology', 'Radiology & Medical Imaging', 'Rehabilitation Therapy',
    'Reproductive Health', 'Rheumatology', 'Social Psychology', 'Surgery', 'Toxicology', 'Transplantation',
    'Tropical Medicine & Parasitology', 'Urology & Nephrology', 'Vascular Medicine', 'Veterinary Medicine',
    'Virology'
]

# Load dataset
try:
    data = pd.read_csv("data/topic_cleaned.csv")
    logger.info("Cleaned data sample: %s", data.head(2).to_dict('records'))
    data.columns = data.columns.str.lower()
    data['key terms'] = data['key terms'].fillna('')
    data['period'] = data['period'].str.strip()
    valid_periods = ['Pre-Covid', 'Covid', 'Post-Covid']
    data = data[data['period'].isin(valid_periods)]
    data_precovid = data[data['period'] == 'Pre-Covid']
    data_covid = data[data['period'] == 'Covid']
    data_postcovid = data[data['period'] == 'Post-Covid']
    logger.info(f"Pre-COVID data: {len(data_precovid)} records")
    logger.info(f"COVID data: {len(data_covid)} records")
    logger.info(f"Post-COVID data: {len(data_postcovid)} records")
except FileNotFoundError:
    logger.error("Error loading topic_cleaned.csv")
    data = pd.DataFrame([
        {"domain": "AIDS & HIV", "period": "Pre-Covid", "topic": 0, "key terms": "aids health incidence late cities subtype disease progression recombination", "prevalence": 0.0947},
        {"domain": "AIDS & HIV", "period": "Covid", "topic": 0, "key terms": "ahisa communicable immunodeficiency prevention treatment diseases asia waiver distancing social distancing acceptability", "prevalence": 0.0498},
        {"domain": "AIDS & HIV", "period": "Post-Covid", "topic": 0, "key terms": "public genes discrimination media challenges", "prevalence": 0.0547},
        {"domain": "Cardiology", "period": "Pre-Covid", "topic": 1, "key terms": "heart surgery recovery", "prevalence": 0.08},
        {"domain": "Cardiology", "period": "Covid", "topic": 1, "key terms": "heart covid impact", "prevalence": 0.06},
        {"domain": "Cardiology", "period": "Post-Covid", "topic": 1, "key terms": "heart recovery post-covid", "prevalence": 0.07},
    ])
    data_precovid = data[data['period'] == 'Pre-Covid']
    data_covid = data[data['period'] == 'Covid']
    data_postcovid = data[data['period'] == 'Post-Covid']

# Utility functions
def filter_by_domain(data, domain):
    filtered = data[data['domain'] == domain]
    logger.debug(f"Filtered data for {domain}: {len(filtered)} records")
    return filtered

def serialize_data(data):
    """Convert list of dicts to a hashable JSON string."""
    return json.dumps(data, sort_keys=True)

@lru_cache(maxsize=128)
def topic_modeling(data_json, num_topics=2):
    data = pd.DataFrame(json.loads(data_json))
    if not data.shape[0]:
        logger.warning("No data for topic modeling")
        return [], None, [], [], {}
    stop_words = set(stopwords.words('english'))
    texts = []
    valid_records = []
    doc_topics = []
    dominant_topics = []
    topic_titles = {}
    
    for _, row in data.iterrows():
        terms = row['key terms']
        if not terms or pd.isna(terms):
            continue
        tokens = [word for word in word_tokenize(terms.lower()) if word not in stop_words and word.isalnum()]
        if len(tokens) < 1:  # Relaxed to 1 token
            continue
        texts.append(tokens)
        valid_records.append(row.to_dict())
    
    if len(texts) < max(1, num_topics):
        logger.warning(f"Insufficient texts ({len(texts)}) for {num_topics} topics")
        return [], None, [], [], {}
    
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    try:
        lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, minimum_probability=0.01, alpha='auto')
        topics = lda.print_topics(num_words=5)
        for topic_id, topic in topics:
            keywords = [word.split('*')[1].strip('"') for word in topic.split(' + ')]
            topic_titles[topic_id] = ' '.join(keywords[:3]).title()
        
        for doc_bow in corpus:
            topic_dist = lda[doc_bow]
            if topic_dist:
                topic_vector = np.zeros(num_topics)
                for topic_id, prob in topic_dist:
                    topic_vector[topic_id] = prob
                doc_topics.append(topic_vector.tolist())
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
                dominant_topics.append(dominant_topic)
            else:
                doc_topics.append(np.zeros(num_topics).tolist())
                dominant_topics.append(-1)
        logger.info(f"Topic modeling completed: {len(valid_records)} records, {num_topics} topics")
    except Exception as e:
        logger.error(f"Error running LDA: {e}")
        return [], None, [], [], {}
    
    return topics, None, dominant_topics, doc_topics, topic_titles

def build_topic_network(data, topic_titles):
    G = nx.Graph()
    topic_groups = data.groupby('topic')
    for topic_id, group in topic_groups:
        key_terms = set()
        for terms in group['key terms']:
            if terms:
                key_terms.update(word_tokenize(terms.lower()))
        G.add_node(topic_id, title=topic_titles.get(topic_id, f"Topic {topic_id}"), key_terms=key_terms)
    
    nodes = list(G.nodes(data=True))
    for i, (node1, data1) in enumerate(nodes):
        for node2, data2 in nodes[i+1:]:
            shared_terms = data1['key_terms'].intersection(data2['key_terms'])
            if shared_terms:
                G.add_edge(node1, node2, weight=len(shared_terms), shared_terms=list(shared_terms))
    logger.debug(f"Topic network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def build_paper_network(valid_records, doc_topics, similarity_threshold=0.2):
    G = nx.Graph()
    if not valid_records or not doc_topics or len(valid_records) != len(doc_topics):
        logger.warning("Invalid input for paper network")
        return G
    for i, record in enumerate(valid_records):
        G.add_node(i, **record)
    for i in range(len(valid_records)):
        for j in range(i + 1, len(valid_records)):
            try:
                vec_i = np.array(doc_topics[i])
                vec_j = np.array(doc_topics[j])
                if np.any(vec_i) and np.any(vec_j):
                    similarity = 1 - cosine(vec_i,vec_j)
                    if similarity > similarity_threshold:
                        G.add_edge(i, j, weight=similarity)
            except:
                continue
    logger.debug(f"Paper network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def detect_communities_girvan_newman(G):
    if len(G.nodes) == 0 or len(G.edges) == 0:
        logger.warning("Empty graph for community detection")
        return {}
    try:
        communities_generator = nx.algorithms.community.girvan_newman(G)
        communities = next(communities_generator)
        result = {idx: list(comm) for idx, comm in enumerate(communities)}
        logger.debug(f"Communities detected: {len(result)}")
        return result
    except:
        logger.error("Error in Girvan-Newman")
        return {}

def calculate_centrality(G):
    if len(G.nodes) == 0:
        logger.warning("Empty graph for centrality")
        return {}, {}, {}
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
    except:
        eigenvector = {node: 0.0 for node in G.nodes}
    logger.debug("Centrality calculated")
    return degree, betweenness, eigenvector

def prepare_network_data(G, communities, degree, betweenness, eigenvector, is_topic_network=False):
    nodes = []
    for node in G.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        title = node_data.get('title', f"Topic {node_id}") if is_topic_network else node_data.get('key terms', f"Record {node_id}")[:50]
        community_id = -1
        for comm_id, members in communities.items():
            if node_id in members:
                community_id = comm_id
                break
        nodes.append({
            "id": str(node_id),
            "title": title.replace('"', "'").replace('\n', ' '),
            "community": community_id,
            "degree": float(degree.get(node_id, 0)),
            "betweenness": float(betweenness.get(node_id, 0)),
            "eigenvector": float(eigenvector.get(node_id, 0))
        })
    edges = []
    for u, v, d in G.edges(data=True):
        weight = float(d.get("weight", 0.5))
        if weight > 0:
            edges.append({
                "source": str(u),
                "target": str(v),
                "weight": weight,
                "shared_terms": d.get("shared_terms", []) if is_topic_network else []
            })
    logger.debug(f"Network data prepared: {len(nodes)} nodes, {len(edges)} edges")
    return {"nodes": nodes, "links": edges}

# Routes
@app.route('/')
def index():
    logger.info("Serving index.html")
    return app.send_static_file('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/api/domains', methods=['GET'])
def get_domains():
    logger.info("Serving domains")
    return jsonify(domains)

@app.route('/api/top_domains', methods=['GET'])
def get_top_domains():
    logger.info("Fetching top 5 domains by prevalence")
    try:
        top_domains = (
            data.groupby('domain')['prevalence']
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
            .to_dict('records')
        )
        logger.debug(f"Top domains: {top_domains}")
        return jsonify(top_domains)
    except Exception as e:
        logger.error(f"Error fetching top domains: {e}")
        return jsonify({"error": "Failed to fetch top domains"}), 500

@app.route('/api/analyze', methods=['GET'])
def analyze():
    domain = request.args.get('domain', domains[0])
    num_topics = max(1, min(10, int(request.args.get('num_topics', 2))))
    logger.info(f"Analyzing domain: {domain}, num_topics: {num_topics}")
    
    pre_data = filter_by_domain(data_precovid, domain)
    covid_data = filter_by_domain(data_covid, domain)
    post_data = filter_by_domain(data_postcovid, domain)

    # Topic modeling
    pre_topics, _, pre_dominant_topics, pre_doc_topics, pre_topic_titles = topic_modeling(serialize_data(pre_data.to_dict('records')), num_topics)
    covid_topics, _, covid_dominant_topics, covid_doc_topics, covid_topic_titles = topic_modeling(serialize_data(covid_data.to_dict('records')), num_topics)
    post_topics, _, post_dominant_topics, post_doc_topics, post_topic_titles = topic_modeling(serialize_data(post_data.to_dict('records')), num_topics)

    # Topic networks
    pre_topic_G = build_topic_network(pre_data, pre_topic_titles)
    covid_topic_G = build_topic_network(covid_data, covid_topic_titles)
    post_topic_G = build_topic_network(post_data, post_topic_titles)

    pre_topic_communities = detect_communities_girvan_newman(pre_topic_G)
    covid_topic_communities = detect_communities_girvan_newman(covid_topic_G)
    post_topic_communities = detect_communities_girvan_newman(post_topic_G)

    pre_topic_degree, pre_topic_betweenness, pre_topic_eigenvector = calculate_centrality(pre_topic_G)
    covid_topic_degree, covid_topic_betweenness, covid_topic_eigenvector = calculate_centrality(covid_topic_G)
    post_topic_degree, post_topic_betweenness, post_topic_eigenvector = calculate_centrality(post_topic_G)

    # Paper networks
    pre_paper_G = build_paper_network(pre_data.to_dict('records'), pre_doc_topics, similarity_threshold=0.2)
    covid_paper_G = build_paper_network(covid_data.to_dict('records'), covid_doc_topics, similarity_threshold=0.2)
    post_paper_G = build_paper_network(post_data.to_dict('records'), post_doc_topics, similarity_threshold=0.2)

    pre_paper_communities = detect_communities_girvan_newman(pre_paper_G)
    covid_paper_communities = detect_communities_girvan_newman(covid_paper_G)
    post_paper_communities = detect_communities_girvan_newman(post_paper_G)

    pre_paper_degree, pre_paper_betweenness, pre_paper_eigenvector = calculate_centrality(pre_paper_G)
    covid_paper_degree, covid_paper_betweenness, covid_paper_eigenvector = calculate_centrality(covid_paper_G)
    post_paper_degree, post_paper_betweenness, post_paper_eigenvector = calculate_centrality(post_paper_G)

    # Prepare data
    pre_topic_network_data = prepare_network_data(pre_topic_G, pre_topic_communities, pre_topic_degree, pre_topic_betweenness, pre_topic_eigenvector, is_topic_network=True)
    covid_topic_network_data = prepare_network_data(covid_topic_G, covid_topic_communities, covid_topic_degree, covid_topic_betweenness, covid_topic_eigenvector, is_topic_network=True)
    post_topic_network_data = prepare_network_data(post_topic_G, post_topic_communities, post_topic_degree, post_topic_betweenness, post_topic_eigenvector, is_topic_network=True)

    pre_paper_network_data = prepare_network_data(pre_paper_G, pre_paper_communities, pre_paper_degree, pre_paper_betweenness, pre_paper_eigenvector)
    covid_paper_network_data = prepare_network_data(covid_paper_G, covid_paper_communities, covid_paper_degree, covid_paper_betweenness, covid_paper_eigenvector)
    post_paper_network_data = prepare_network_data(post_paper_G, post_paper_communities, post_paper_degree, post_paper_betweenness, post_paper_eigenvector)

    # Trend analysis
    trend_data = {
        "pre": pre_data.groupby('topic')['prevalence'].mean().to_dict(),
        "covid": covid_data.groupby('topic')['prevalence'].mean().to_dict(),
        "post": post_data.groupby('topic')['prevalence'].mean().to_dict()
    }

    # Word cloud data
    def prepare_wordcloud_data(topics):
        words = []
        for topic_id, topic in topics:
            for term in topic.split(' + '):
                weight, word = term.split('*')
                words.append([word.strip('"'), float(weight) * 100])
        return words[:20]

    response = {
        "pre_topics": pre_topics,
        "covid_topics": covid_topics,
        "post_topics": post_topics,
        "pre_topic_network": pre_topic_network_data,
        "covid_topic_network": covid_topic_network_data,
        "post_topic_network": post_topic_network_data,
        "pre_paper_network": pre_paper_network_data,
        "covid_paper_network": covid_paper_network_data,
        "post_paper_network": post_paper_network_data,
        "pre_topic_communities": pre_topic_communities,
        "covid_topic_communities": covid_topic_communities,
        "post_topic_communities": post_topic_communities,
        "pre_paper_communities": pre_paper_communities,
        "covid_paper_communities": covid_paper_communities,
        "post_paper_communities": post_paper_communities,
        "pre_topic_centrality": {
            "degree": pre_topic_degree,
            "betweenness": pre_topic_betweenness,
            "eigenvector": pre_topic_eigenvector
        },
        "covid_topic_centrality": {
            "degree": covid_topic_degree,
            "betweenness": covid_topic_betweenness,
            "eigenvector": covid_topic_eigenvector
        },
        "post_topic_centrality": {
            "degree": post_topic_degree,
            "betweenness": post_topic_betweenness,
            "eigenvector": post_topic_eigenvector
        },
        "pre_paper_centrality": {
            "degree": pre_paper_degree,
            "betweenness": pre_paper_betweenness,
            "eigenvector": pre_paper_eigenvector
        },
        "covid_paper_centrality": {
            "degree": covid_paper_degree,
            "betweenness": covid_paper_betweenness,
            "eigenvector": covid_paper_eigenvector
        },
        "post_paper_centrality": {
            "degree": post_paper_degree,
            "betweenness": post_paper_betweenness,
            "eigenvector": post_paper_eigenvector
        },
        "trend_data": trend_data,
        "pre_wordcloud": prepare_wordcloud_data(pre_topics),
        "covid_wordcloud": prepare_wordcloud_data(covid_topics),
        "post_wordcloud": prepare_wordcloud_data(post_topics)
    }
    logger.info(f"Analyze response: {len(response['pre_topics'])} pre-topics, {len(response['pre_topic_network']['nodes'])} pre-topic nodes")
    return jsonify(response)

@app.route('/api/export_pdf', methods=['GET'])
def export_pdf():
    domain = request.args.get('domain', domains[0])
    return jsonify({"message": "PDF export triggered (implement on frontend)"})

if __name__ == '__main__':
    app.run(debug=True)