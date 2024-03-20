
def generate_gaussian_elements(element_set, window_matrix):
    return 'a'


def calculate_gaussian_embedding_for_dataset(dataset_dict, window_matrix):
    gaussian_dict = {}
    for label, _ in dataset_dict.items():
        gaussian_dict[label] = []
        for element_set in dataset_dict[label]:
            gaussian_embedded = generate_gaussian_elements(element_set, window_matrix)
            gaussian_dict[label].append(gaussian_embedded)
    return gaussian_dict

def calculate_gaussian_embedding_for_gaussian_model(gaussian_dict):
    #有几种距离，这里可以直接用LogEuclidean, 其他的距离先不实现了。
    return []