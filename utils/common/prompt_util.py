import generation.configs as configs
import random
import json

def random_context(ref_samples):
    index = random.randint(0, len(ref_samples) - 1)
    content = ref_samples[index]
    max_context = content[:min(len(content), configs.MAX_SIZE_CONTEXT)]
    return max_context

def make_example_context_str(ref_samples):
    picked = []
    sample_context = ""
    sample_lenght = len(ref_samples)
    for _ in range(sample_lenght):
        index = random.randint(0, len(ref_samples) - 1)
        
        if _ + 1 > configs.MAX_SAMPLE_NUMBER:
            break

        elif (len(sample_context + str(ref_samples[index])) < configs.MAX_SIZE_CONTENT_REQUEST):
            if  index not in picked:
                sample_context += f'''\n\tví dụ {len(picked) + 1}: {str(ref_samples[index])}'''
                picked.append(index)
 
        else:
            break
    return sample_context


def get_sample_context(ref_samples):
    # Get vung_bien samples data
    picked = []
    sample_context = ""
    sample_lenght = len(ref_samples)
    for _ in range(sample_lload_5_report_ref_samplesenght):
        index = random.randint(0, len(ref_samples) - 1)
        
        if (len(sample_context + ref_samples[index]) < configs.MAX_SIZE_CONTENT_REQUEST):
            if  index not in picked:
                sample_context += f'''\nví dụ {len(picked) + 1}: {ref_samples[index]}'''
                picked.append(index)
        
        elif _ + 1 > configs.MAX_SAMPLE_NUMBER:
            break
 
        else:
            break
    return sample_context


def load_sample(sample_location_path, separate_char = '--------'):
    samples = []
    with open(sample_location_path, 'r') as f:
        lines = f.readlines()
        sample = []
        for line in lines:
            if separate_char not in line:
                sample.append(line)
            else:
                samples.append("".join(sample))
                sample = []
    return samples

def load_contexts(path):
    with open(path, 'r') as f:
        datas  = json.load(f)
    contents = list(set([x['data'] for x in datas]))
    return contents