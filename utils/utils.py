import io
import torch
import streamlit as st

def download_model_button(trained_model, filename="model.pth"):
    buffer = io.BytesIO()
    torch.save(trained_model, buffer)
    buffer.seek(0)
    st.download_button(
        label="Download Trained Model",
        data=buffer,
        file_name=filename,
        mime="application/octet-stream",
    )

def generate_new_graph(G, dict, id_map):
    G_copy = G.clone()
    n_feat = []
    for idx, i in enumerate(G_copy['PRODUCT_OFFERING'].x):
        node_id = G_copy['PRODUCT_OFFERING'].node_id[idx]
        po_key = list(id_map.keys())[list(id_map.values()).index(node_id)]
        last_value = dict[po_key][-1] if isinstance(dict[po_key], list) else None
        if last_value is not None:
            i = i.tolist()
            i[1] = last_value
            n_feat.append(i)
        else:
            i = i.tolist()
            n_feat.append(i)
    G_copy['PRODUCT_OFFERING'].x = torch.tensor(n_feat, dtype=torch.float)
    G_copy.x_dict['PRODUCT_OFFERING'] = torch.tensor(n_feat, dtype=torch.float)
    return G_copy