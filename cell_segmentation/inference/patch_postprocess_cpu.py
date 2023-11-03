# BEGIN: 7d5c9d3f5d6c
import numpy as np
import torch
from einops import rearrange


def worker_function(input_queue, output_queue):
    while True:
        item = queue.get()
        # check for stop
        if item is None:
            break
        # unpack item
        CellDectInstance, predictions, metadata, dataset_config, wsi, overlap = item
        # process item
        cell_dict_wsi, cell_dict_detection, graph_data, processed_patches = process_cell_detection(
            CellDectInstance, predictions, metadata, dataset_config, wsi, overlap
        )
        # put result in output queue
        output_queue.put(
            (cell_dict_wsi, cell_dict_detection, graph_data, processed_patches)
        )



def process_cell_detection(CellDectInstance, predictions, metadata, dataset_config, wsi, overlap):
    instance_types = []
    tokens = []
    cell_dict_wsi = []
    cell_dict_detection = []
    graph_data = {"cell_tokens": [], "positions": [], "contours": []}
    processed_patches = []

    # predictions = self.model.reshape_model_output(predictions_, self.device)
    instance_types, tokens = CellDectInstance.get_cell_predictions_with_tokens(
        predictions, magnification=wsi.metadata["magnification"]
    )

    # unpack each patch from batch
    for idx, (patch_instance_types, patch_metadata) in enumerate(zip(instance_types, metadata)):
        
        # add global patch metadata
        patch_cell_detection = {}
        patch_cell_detection["patch_metadata"] = patch_metadata
        patch_cell_detection["type_map"] = dataset_config["nuclei_types"]
        
        processed_patches.append(
            f"{patch_metadata['row']}_{patch_metadata['col']}"
        )

        # calculate coordinate on highest magnifications
        wsi_scaling_factor = wsi.metadata["downsampling"]
        patch_size = wsi.metadata["patch_size"]
        x_global = int(
            patch_metadata["row"] * patch_size * wsi_scaling_factor
            - (patch_metadata["row"] + 0.5) * overlap
        )
        y_global = int(
            patch_metadata["col"] * patch_size * wsi_scaling_factor
            - (patch_metadata["col"] + 0.5) * overlap
        )

        # extract cell information
        for cell in patch_instance_types.values():
            if cell["type"] == dataset_config["nuclei_types"]["Background"]:
                continue
            offset_global = np.array([x_global, y_global])
            centroid_global = cell["centroid"] + np.flip(offset_global)
            contour_global = cell["contour"] + np.flip(offset_global)
            bbox_global = cell["bbox"] + offset_global
            cell_dict = {
                "bbox": bbox_global.tolist(),
                "centroid": centroid_global.tolist(),
                "contour": contour_global.tolist(),
                "type_prob": cell["type_prob"],
                "type": cell["type"],
                "patch_coordinates": [patch_metadata["row"], patch_metadata["col"]],
                "cell_status": get_cell_position_marging(cell["bbox"], 1024, 64),
                "offset_global": offset_global.tolist(),
            }
            cell_detection = {
                "bbox": bbox_global.tolist(),
                "centroid": centroid_global.tolist(),
                "type": cell["type"],
            }
            if np.max(cell["bbox"]) == 1024 or np.min(cell["bbox"]) == 0:
                position = get_cell_position(cell["bbox"], 1024)
                cell_dict["edge_position"] = True
                cell_dict["edge_information"] = {}
                cell_dict["edge_information"]["position"] = position
                cell_dict["edge_information"]["edge_patches"] = get_edge_patch(
                    position, patch_metadata["row"], patch_metadata["col"]
                )
            else:
                cell_dict["edge_position"] = False

            cell_dict_wsi.append(cell_dict)
            cell_dict_detection.append(cell_detection)

            # get the cell token
            bb_index = cell["bbox"] / 1024
            bb_index[0, :] = np.floor(bb_index[0, :])
            bb_index[1, :] = np.ceil(bb_index[1, :])
            bb_index = bb_index.astype(np.uint8)
            cell_token = tokens[
                idx,
                bb_index[0, 1] : bb_index[1, 1],
                bb_index[0, 0] : bb_index[1, 0],
                :,
            ]
            cell_token = torch.mean(rearrange(cell_token, "H W D -> (H W) D"), dim=0)

            graph_data["cell_tokens"].append(cell_token)
            graph_data["positions"].append(torch.Tensor(centroid_global))
            graph_data["contours"].append(torch.Tensor(contour_global))

    return cell_dict_wsi, cell_dict_detection, graph_data, processed_patches
