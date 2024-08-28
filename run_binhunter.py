import joblib
import argparse
from binhunter_train.gcn import GCN

def validate_graphs(sub_graphs, graphs, graph_labels, binary_paths, edges):
    subs = []
    gs = []
    ls = []
    bins = []
    edgs = []
    for idx in range(len(sub_graphs)):
        if sub_graphs[idx] is not None and graphs[idx] is not None :
            if len(sub_graphs[idx].nodes()) and len(graphs[idx].nodes()) :
                subs.append(sub_graphs[idx])
                gs.append(graphs[idx])
                ls.append(graph_labels[idx])
                bins.append(binary_paths[idx])
                edgs.append(edges[idx])
    return subs , gs , ls , bins,  edgs


def main():
    
    parser = argparse.ArgumentParser(description="Process the CWE-type argument.")
    parser.add_argument('CWE_type', type=str, help='The CWE type to process')
    
    args = parser.parse_args()
    
    # Access the CWE-type argument
    cwe_type = args.CWE_type
    
    # Print the CWE-type
    print(f"Processing for CWE-type: {cwe_type}")

    pdg_subgraphs = joblib.load('data/'+cwe_type+'/binhunter_garphs.pkl')
    pdgs = joblib.load('data/'+cwe_type+'/binhunter_ddg_graphs_simple.pkl')
    labels = joblib.load('data/'+cwe_type+'/binhunter_labels.pkl')
    bin_paths = joblib.load('data/'+cwe_type+'/binary_paths.pkl')
    allowed_edges = joblib.load('data/'+cwe_type+'/permitted_edges.pkl')
    params = joblib.load('data/'+cwe_type+'/params.pkl')



    assert len(pdg_subgraphs) == len(pdgs) == len(labels) == len(bin_paths) == len(allowed_edges)

    npdg_subgraphs, npdgs, nlabels, nbin_paths,  nallowed_edges = validate_graphs(pdg_subgraphs, pdgs, labels, bin_paths,  allowed_edges)
            
    #breakpoint()
    gcn_model = GCN(npdg_subgraphs, npdgs, cwe_type, nlabels, nbin_paths,params, nallowed_edges, None)
    
    gcn_model.train_gcn()


if __name__ == "__main__":
    main()