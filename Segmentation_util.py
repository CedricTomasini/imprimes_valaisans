#  @Author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# Utility functions related to characters seen as blobs (connected components) and group by words and lines


# Among other things, generate a list of all boundary box for all blob,
# discriminate letters VS non-letter blobs
# and provide a tool to find the size of the largest letters of each font size in the document.
# In the second part, build line with the word informations
# In the third part, reorder the lines in the usual western reading ordering.


# Meta-parameters (you can try to change them if needed)
# This parameters are magic numbers used to perform segmenting. They often define validity ratio or areas.
#
# min_area_med_ratio (is_valid_box) : minimum ratio (current box area)/(median box area) for the letter box to be valid.
# max_area_page_ratio (is_valid_box) : maximum ratio (current box area)/(full page area) for the letter box to be valid.
# height_length_max_ratio (is_valid_box) : maximum ratio (box height)/(box width) for the letter box to be valid.
# width_length_max_ratio (is_valid_box) : maximum ratio (box width)/(box height) for the letter box to be valid.
#
# size and type of the smoothing filter (get_valid_heights_hist_maxima) : format du filtre pour lisser histogramme des hauteurs.
#
# restriction_factor (merge_close_peaks) : say to which point a peak is close enough to an other one to be merged.
#
# page_size_max_ratio (is_black_separator) : alias for 1/max_area_page_ratio, but in another function. Supposed to be the same.
# narrow_ratio (is_black_separator) : pseudo-alias for height_length_max_ratio. Minimum ratio to be considered a vertical sep.
#
# max_relative_space(build_line) : Initial value x, so that x*(height of line) is the furthest the next word can be.
# max_deviation_space (build_line) : when further in the current line processing, x =(median space)*(1+ max_deviation_space).
# ray_border_cropping (build_line) : how much we cut on each side of the ray that beams from the current line to find next word.
# max_height_ratio (build_line) : the next word height must be (between max_height_ratio and 1/max_height_ratio) *(line height)
#
# width_height_max_ratio (create_segment_file) : pseudo-alias for width_length_max_ratio. Minimum ratio to be a horizontal sep.
#
# lower_limit_margin_ratio (process_column) : allow an overlap when looking for lower neighbors. Linked to ray_border_cropping

from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import scipy.signal
import scipy.stats
from tqdm import tqdm
import os
import IV_util


#### Color equalizing

def is_white(im_array):
    """
    Tell if pixels of an image are white or not.
    im_array : the image represented as a numpy array in RGB (width*height*3)
    Return : a numpy mask of size width*height
    """
    return (im_array[:,:,0] == 255) & (im_array[:,:,1] == 255) & (im_array[:,:,2] == 255)

def is_almost_white(im_array,threshold = 200):
    """
    Tell if pixels of an image are white or not.
    im_array : the image represented as a numpy array in RGB (width*height*3)
    threshold: the threshold above which a pixel is considered as white
    Return : a numpy mask of size width*height
    """
    return (im_array[:,:,0] > threshold) & (im_array[:,:,1] > threshold) & (im_array[:,:,2] > threshold)


def generate_boundary_blurred_image(im, filter_size = 21):
    """
    Generate an image where all part close to white are replaced by the mean color of the image
    and the boundary between the document and the background is blurred.
    This adapt the image better for Kraken binarization.
    im : the PIL Image object to process
    filter_size : the size of the boundary filter. 21 was sufficient
    Return : a PIL Image object
    
    
    Génère une image plus adaptée à la binarization avec Kraken, en remplaçant les parties très claires
    de l'image par la couleur moyenne de cette image, et en floutant la zone séparant l'image du fond blanc.
    
    Combiner le remplacement du blanc par la couleur moyenne et le floutage de la frontière permet
    de diminuer la quantité de floutage nécessaire, et donc de gagner du temps et de diminuer la taille
    de la marge "perdue" dans l'opération.
    
    Un effet secondaire bienvenu de cette fonction est qu'elle est parfois capable d'aplanir l'intérieur
    de l'image et ainsi de diminuer le bruit, sans jamais altérer le texte, puisque celui-ci est sombre.
    """
    
    im_array = np.array(im)
    
    #Find boundaries
    im_shift_right = np.roll(im_array,shift = 1,axis = 1)
    im_shift_right[:,0] = im_shift_right[:,1] # On pad en recopiant la valeur juste à côté

    im_shift_left = np.roll(im_array,shift = -1,axis = 1)
    im_shift_left[:,-1] = im_shift_left[:,-2]

    im_shift_down = np.roll(im_array,shift = 1, axis = 0)
    im_shift_down[0,:] = im_shift_down[1,:]

    im_shift_up = np.roll(im_array,shift = -1, axis = 0)
    im_shift_up[-1,:] = im_shift_up[-2,:]

    boundary_mask = is_white(im_array) & ~(is_white(im_shift_left) & is_white(im_shift_right) &
                                       is_white(im_shift_down) & is_white(im_shift_up))
    
    #Generate the filtering zone
    filtre = np.ones((filter_size,filter_size))/np.square(filter_size)
    convolve_mask = scipy.signal.convolve(boundary_mask,filtre,mode="same")*np.square(filter_size) >= 1
    extended_convolve_mask = np.dstack((convolve_mask,convolve_mask,convolve_mask))
    
    
    # Replace the almost white part of the image with the mean color of the image
    # We use almost white because some part of the boundary are only partially white
    # (Probably because of some jpeg compression)
    rgb_mean = np.array([im_array[:,:,0].mean(),im_array[:,:,1].mean(),im_array[:,:,2].mean()])
    im_array[is_almost_white(im_array),:] = rgb_mean.astype("uint8")
    
    
    #Create a filtered version of the image
    convolved_im_array = np.dstack((scipy.signal.convolve(im_array[:,:,0],filtre,mode="same"),
                                 scipy.signal.convolve(im_array[:,:,1],filtre,mode="same"),
                                 scipy.signal.convolve(im_array[:,:,2],filtre,mode="same")))
    
    #Generating the final image
    mixed_array = (convolved_im_array*extended_convolve_mask + im_array*(1-extended_convolve_mask)).astype("uint8")
    bb_im = Image.fromarray(mixed_array)
    
    return bb_im



##### Boxes and words generation

### Private functions (Utility functions for the public functions. They should not be used outside of this file)

def is_valid_box(box_stat, median ,im):
    """
    Decide wether a component detected is a letter or not.
    Components who are not a letter include line separator, decoration
    or the big component that cover most of the page.
    
    box_stat : The component to be judged
    median : The array containing all the boxes.
    im : The PIL Image.
    Return boolean
    
    Condition d'aire minimale.
    Condition d'aire maximale.
    Condition de forme pas trop étroite.
    Pas de condition sur la hauteur relative de l'objet détecté, au cas où une image est juste un tout petit fragment
    de texte sans marges. (Pourrait arriver si des images déjà découpées avec ScanTailor sont traitées.)
    Pas de condition de nombre minimum de boîtes d'une certaine hauteur.
    """
    
    im_width, im_height = im.size
    
    min_area_med_ratio = 0.2
    max_area_page_ratio = 0.5
    height_length_max_ratio = 8 #Etait 10 auparavant, mais semble un peu élevé
    width_length_max_ratio = 4
    
    x = box_stat[0] #anchor x
    y = box_stat[1] #anchor y
    w = box_stat[2] #width
    h = box_stat[3] #height
    a = box_stat[4] #area
    
    # Area that are too small are not included. We define "too small" as "less than x% of the median area".
    cond_median_area = a > median*min_area_med_ratio
    
    # Box that are bigger than half the page are not considered.
    cond_im_area = a < im_width*im_height*max_area_page_ratio
    
    # Box whose shape is too narrow are not considered.
    cond_shape = w < width_length_max_ratio*h and h < height_length_max_ratio*w
    
    return cond_median_area and cond_im_area and cond_shape


def generate_spherical_1D_filter(size):
    """
    Generate a discrete circle-shaped 1D filter of odd size
    Return a list of length size
    
    Le filtre en forme de demi-cercle a été choisi car il présente une forme de plateau
    """
    
    assert size%2==1
    
    x = np.linspace(-1,1,num = size+2)[1:-1]
    
    y = np.sqrt(1-np.square(x))
    
    return y


def get_valid_heights_hist_maxima(valid_boxes_list,verbose = False):
    """
    valid_boxes_list : the stats obtained with cv2.connectedComponentsWithStats
    Return peaks list ; dictionary of find_peaks function parameters (currently empty and not used)
    
    Filter a box list to keep only valid boxes, then generate the histogramm of heights,
    smooth it and find all local maxima.
    
    On dresse l'histogramme des hauteurs de boîte de détection et on le lisse avec notre filtre,
    puis on retourne les pics de cet histogramme
    Empiriquement, le meilleur filtre trouvé jusqu'à présent est un filtre en demi-disque de taille 7
    """
    
    valid_boxes_heights = valid_boxes_list[:,3]
    
    heights_hist = np.bincount(valid_boxes_heights)
    
    #filtre = np.array([1,1,1,1,1])
    #filtre = generate_gaussian_1D_filter(9,90)
    filtre = generate_spherical_1D_filter(7)
    
    smoothed_heights_hist = np.convolve(heights_hist,filtre,mode = "same")
    
    if verbose:
        n_bins = max(valid_boxes_heights)+1 #One bin per value, including 0, til max value.
        
        plt.hist(valid_boxes_heights,bins = n_bins)
        plt.show()
           
        plt.bar(range(len(heights_hist)),smoothed_heights_hist)
        plt.show()
    
        print("All peaks", list(scipy.signal.find_peaks(heights_hist)[0]))
        print("Peaks after smoothing",scipy.signal.find_peaks(smoothed_heights_hist)[0])
        
    return scipy.signal.find_peaks(smoothed_heights_hist)


def merge_close_peaks(peaks_list):
    """
    We define too close peaks as peaks whose difference is less than 10% of their value.
    If multiple peaks successively match the difference, they are all pairwise converted
    The function proceed recursively until there is no more change to make.
    
    Example of step-by-step processing:
    [51,52,53,100,200,201] -> [51.5,52.5,100,200.5] -> [52,100,200.5] --> [52,100,200] (int conversion)
    """
    peaks_list = np.array(sorted(peaks_list))
    
    restriction_factor = 10
    
    inter_means = (peaks_list[1:] + peaks_list[:-1])/2
    inter_diffs = peaks_list[1:] - peaks_list[:-1]
    merge_with_next_mask = inter_diffs*restriction_factor < inter_means
    
    if merge_with_next_mask.sum() > 0:
        
        new_peaks = []
        for i in range(len(merge_with_next_mask)):
            if merge_with_next_mask[i]:
                new_peaks.append((peaks_list[i] + peaks_list[i+1])/2)
            else:
                # We append only if there was no merge in the previous step
                # (in the latter case, the value is "contained" in the previous merge therefore not kept here)
                if i == 0 or not merge_with_next_mask[i-1]:
                    new_peaks.append(peaks_list[i])

                # If the two last peaks are not merged, we still need to add the last one to the list
                if i == len(merge_with_next_mask)-1:
                    new_peaks.append(peaks_list[i+1])
        
        return merge_close_peaks(new_peaks)
    
    else:
        return peaks_list.astype(int)
    
    
def find_neighbors(row, df):
    """
    Find for the given connected component stored in row the index of its neighbors in df
    
    row: row whose neighbors we are looking for.
    df : the dataframe in which to search the neigbhors.
    Return : a list of size 2
    """
    
    #We define possible neighbors as boxes which share the horizontal plane where our current box sits.
    possible_neighbors = df[(df["antipod_y"] > row["anchor_y"]) & (df["anchor_y"] < row["antipod_y"])]
    
    possible_left_neighbors = possible_neighbors[possible_neighbors["anchor_x"] < row["anchor_x"]]
    possible_right_neighbors = possible_neighbors[possible_neighbors["anchor_x"] > row["anchor_x"]]
    
    if not possible_left_neighbors.empty:
        
        #Find the closest neighbor
        candidate_left_neighbor =\
        possible_left_neighbors.iloc[np.argmin(np.abs(possible_left_neighbors["anchor_x"] - row["anchor_x"]))]

        #Verify if the closest neighbor is close enough to be part of the same word.
        #We define "close enough" as "closer than the width of the current box"
        if candidate_left_neighbor["antipod_x"] > row["anchor_x"] - row["width"]:
            left_neighbor = candidate_left_neighbor.name #Get index of the neighbor
        else:
            left_neighbor = None
    else:
        left_neighbor = None
        
    if not possible_right_neighbors.empty:
        
        candidate_right_neighbor =\
        possible_right_neighbors.iloc[np.argmin(np.abs(possible_right_neighbors["anchor_x"] - row["anchor_x"]))]
        
        if candidate_right_neighbor["anchor_x"] < row["antipod_x"] + row["width"]:
            right_neighbor = candidate_right_neighbor.name
        else:
            right_neighbor = None
    else:
        right_neighbor = None
        
    return [left_neighbor, right_neighbor]


def fill_word_right(word_num, current_char, right_neighbor, df):
    """
    Give number to words in a recursive manner, from left to right. Update df inplace
    
    word_num: the number of the current word.
    current_char : the index of the last box to have been labelled with this number.
    right_neighbor : the index of the right neibor
    df : the dataFrame to use and modify.
    Return None
    """
    
    if right_neighbor:
        
        #This condition should always be met, but we keep it
        #in order to never enter an endless recursive loop,
        #no matter what strange behavior may happen.
        if not df.loc[right_neighbor,"word"]:
        
            if df.loc[right_neighbor,"left_neighbor"] == current_char:
                df.loc[right_neighbor,"word"] = word_num

                fill_word_right(word_num,right_neighbor,df.loc[right_neighbor,"right_neighbor"],df)

                
def fill_word_left(word_num, current_char, left_neighbor, df):
    """
    Give number to words in a recursive manner, from right to left. Update df inplace
    This function exists because the first char we met in the df may not be the first character of the word
    
    word_num: the number of the current word.
    current_char : the index of the last box to have been labelled with this number.
    right_neighbor : the index of the right neibor
    df : the dataFrame to use and modify.
    Return None
    """
    
    if left_neighbor:
        
        if not df.loc[left_neighbor, "word"]:
            
            if df.loc[left_neighbor, "right_neighbor"] == current_char:
                df.loc[left_neighbor, "word"] = word_num
                
                fill_word_left(word_num, left_neighbor, df.loc[left_neighbor,"left_neighbor"],df)
    
    
def has_no_jump(bigram, peaks_groundtruth):
    """
    Tell if the two components of the bigram are same or successive in the sequence of valid peaks or not
    For exemple, if groundtruth = [1,2,3], [1,1] or [2,3] have no jump but [1,3] has a jump.
    
    bigram : the bigram to judge
    peaks_groundtruth : the list of valid peaks
    Return boolean
    """

    assert len(bigram) == 2
    
    if len(set(bigram)) == 1:
        return True
    
    sorted_groundtruth = sorted(peaks_groundtruth)
    
    sorted_peaks = sorted(list(bigram))
    
    begin = sorted_groundtruth.index(sorted_peaks[0])
    end = begin+len(sorted_peaks)
    
    return sorted_peaks == sorted_groundtruth[begin:end]


def count_nondirectionnal_bigrams(l,global_dict):
    """
    Update the global_dict to count each bigram, but not considering the order : (1,2) is the same as (2,1)
    
    l : the list of connected components heights for a given word
    global_dict : the dictionary to update
    Return None
    """
    
    for i in range(len(l)-1):
        bigram = tuple(sorted([l[i],l[i+1]]))
        
        if bigram not in global_dict:
            global_dict[bigram] = 1
        else:
            global_dict[bigram] += 1
            
###



### Public functions

def generate_all_ConnectedComponents_stats(bw_pil_image):
    """
    For a given bw image, return the stats of boundary boxes of all connected components detected in the image
    
    bw_pil_image : the black-and-white PIL image object
    Return : a list of lists. Each sublist l consists of five elements:
    l[0] = x of upper-left corner of the component
    l[1] = y of the upper-lef corner of the component
    l[2] = width of the component
    l[3] = height of the component
    l[4] = area of the component
    """
    
    # Image creation, inversion and search for connected components
    
    im = bw_pil_image

    im_array = np.array(im)

    reverse_im_array = ((im_array == 0)*255).astype("uint8")
    reverse_im = Image.fromarray(reverse_im_array)

    cc_output = cv2.connectedComponentsWithStats(np.uint8(reverse_im))
    
    
    # Box stats
    
    box_stats = cc_output[2]
    
    return box_stats
    

def generate_invalid_box_stats_df(box_stats,im):
    """
    Generate a dataframe of the same form as box_stats only for non-valid boxes (boxes that are probably not a character),
    but with two extra columns antipod_x and antipod_y for the coordinates of the lower-right corner.
    This function is complementary to generate_wordy_box_stats_df.
    
    box_stats : a list of lists as returned by generate_all_ConnectedComponents_stats_df
    im : the black and white PIL image
    Return : a Pandas DataFrame
    """
    
    valid_box_mask = np.apply_along_axis(lambda b : is_valid_box(b,np.median(box_stats),im),axis=1,arr=box_stats)
    invalid_box_stats = box_stats[~valid_box_mask]
    
    invalid_box_df = pd.DataFrame(invalid_box_stats, columns = ("anchor_x","anchor_y","width","height","area"))
    invalid_box_df["antipod_x"] = invalid_box_df["anchor_x"] + invalid_box_df["width"]
    invalid_box_df["antipod_y"] = invalid_box_df["anchor_y"] + invalid_box_df["height"]
    
    return invalid_box_df


def generate_wordy_box_stats_df(box_stats,im,return_peaks = False):
    """
    Generate a dataframe of the same form as box_stats only for valid boxes (boxes that are probably a character).
    The dataframe contains also informations about the neighbors of the box and the word it belong to.
    Extra columns are:
    antipod_x and antipod_y : the coordinates of the lower-right corner of the box
    closest_peak : among the peaks found in the histogramm, the closest to the height of the box (serve as a bucket)
    left_neighbor and right_neighbor : the index of the neighbors of the box, if any. Else None
    word : the word number of the box
    
    Generate also the peaks list that need also to be used later
    
    box_stats : a list of lists as returned by generate_all_ConnectedComponents_stats_df
    im : the black and white PIL image
    Return : a Pandas DataFrame
    """
    
    # Filtering of valid boxes
    
    valid_box_mask = np.apply_along_axis(lambda b : is_valid_box(b,np.median(box_stats),im),axis=1,arr=box_stats)
    valid_box_stats = box_stats[valid_box_mask]
    
    
    # Peaks extraction
    
    smooth_peaks, _ = get_valid_heights_hist_maxima(valid_box_stats)
    
    
    # Close peaks merging
    
    valid_smooth_peaks = merge_close_peaks(smooth_peaks)
    
    
    # Box dataframe creation 
    
    box_df = pd.DataFrame(valid_box_stats, columns = ("anchor_x","anchor_y","width","height","area"))
    box_df["antipod_x"] = box_df["anchor_x"] + box_df["width"]
    box_df["antipod_y"] = box_df["anchor_y"] + box_df["height"]

    box_df["closest_peak"] = box_df["height"].apply(
        lambda h : valid_smooth_peaks[np.argmin(np.abs(valid_smooth_peaks - h))])
    
    
    # Neighbors finding
    
    box_df["left_neighbor"], box_df["right_neighbor"] = zip(*box_df.apply(lambda r : find_neighbors(r,box_df), axis = 1))
    
    
    # Box linking through word relation
    
    box_df["word"] = None
    
    w_n = 0
    for i, _ in box_df.iterrows():
        if not box_df.loc[i,"word"]:
            box_df.loc[i,"word"] = w_n

            fill_word_right(w_n,i,box_df.loc[i,"right_neighbor"],box_df)
            fill_word_left(w_n,i,box_df.loc[i,"left_neighbor"],box_df)

            w_n += 1
    
    if return_peaks:
        return box_df, valid_smooth_peaks
    else:
        return box_df


def find_optimal_peaks(box_df, valid_smooth_peaks):
    """
    Group the peaks depending on the words in which they are contained and return a peak list
    where each peak is supposed to represent a font size category (max size, not mean size,
    because the boxes generated by kraken segmentation surround the whole text)
    """
    
    # Creation of a word dataframe
    
    words_group =  box_df.groupby("word")["closest_peak"]
    words_char_height = words_group.apply(list)
    words_peaks = words_group.apply(frozenset)
    words_length = words_group.count()
    words_df = pd.DataFrame(zip(words_char_height,words_peaks,words_length),
                            columns = ("char_heights","peaks","length"))

    #We keep only "words" of length 3 or more
    valid_words_df = words_df[words_df["length"] > 2]
    
    
    # Bigrams count
    
    bigram_dict = {}
    valid_words_df.char_heights.apply(lambda l : count_nondirectionnal_bigrams(l,bigram_dict))

    bigram_df = pd.DataFrame(bigram_dict.items(), columns = ("bigram","number"))

    valid_bigram_df = bigram_df[bigram_df.bigram.apply(lambda b : has_no_jump(b, valid_smooth_peaks))]
    
    
    # Final peaks grouping
    
    list_of_final_peaks_dict = []
    for bigram in valid_bigram_df.sort_values("number", ascending = False)["bigram"]:

        el1 = bigram[0]
        el2 = bigram[1]

        present_flag = False

        for i, p in enumerate(list_of_final_peaks_dict):

            cond_anchor = (el1 == p["lowercase_anchor"]) or (el2 == p["lowercase_anchor"])
            cond_group = el1 in p["group"] and el2 in p["group"]

            if cond_anchor or cond_group:
                list_of_final_peaks_dict[i]["group"].add(el1)
                list_of_final_peaks_dict[i]["group"].add(el2)
                present_flag = True

        if not present_flag:
            list_of_final_peaks_dict.append({"lowercase_anchor":min(el1,el2),"group":set([el1,el2])})
    
    
    final_box_sizes = [max(d["group"]) for d in list_of_final_peaks_dict]
    
    return final_box_sizes


##### Line generation

def aggregate_to_next_level(box_df,next_level_name):
    """
    From a dataFrame where each row correspond to a box, and each row has a column next_level_name that indicate
    to which super-box the row belong, generate the dataframe where each super-box has a row.
    
    box_df : the source dataFrame
    next_level_name : the column that store to which super-box each row belong.
    Return : a dataFrame
    """
    
    box_df["temp_length"] = 1
    next_level_group = box_df.groupby(next_level_name)
    
    next_level_df = \
    next_level_group.agg({"anchor_x":min,"anchor_y":min,"antipod_x":max,"antipod_y":max,"temp_length":"count"})
    next_level_df = next_level_df.rename(columns = {"temp_length":"length"})
    box_df.drop(columns = ["temp_length"], inplace = True)
    
    next_level_df["width"] = next_level_df["antipod_x"] - next_level_df["anchor_x"]
    next_level_df["height"] = next_level_df["antipod_y"] - next_level_df["anchor_y"]
    
    return next_level_df


def is_black_separator(row,im):
    """
    For a given probably-not-a-word connected-component box, tell if it is probably a separator or not
    
    row : a row of the boxes dataFrame
    im : the image to be checked
    Return : bool
    """
    
    page_size_max_ratio = 2
    narrow_ratio = 8
    
    not_full_page_cond = row["area"]*page_size_max_ratio < np.prod(im.size)
    vertical_narrow_cond = row["height"] > row["width"]*narrow_ratio
    
    return not_full_page_cond and vertical_narrow_cond


def draw_boxes(im_path, df, save_path):
    """
    Draw all boxes on the related image
    im_path : the path of the image to draw on
    df : the dataFrame containing all the boxes
    save_path : the path where to save the new image
    """
    
    im = Image.open(im_path)
    draw = ImageDraw.Draw(im)

    for _, box in df.iterrows():

        x1 = box["anchor_x"]
        y1 = box["anchor_y"]
        x2 = box["antipod_x"]
        y2 = box["antipod_y"]

        tl = (x1,y1)
        tr = (x2,y1)
        bl = (x1,y2)
        br = (x2,y2)
        
        if "valid" in df.columns:
            if box["valid"]:
                draw.line([tl,tr,br,bl,tl], fill = (0,0,255), width = 2)
            else:
                draw.line([tl,tr,br,bl,tl], fill = (255,0,0), width = 2)
        else:
            draw.line([tl,tr,br,bl,tl], fill = (255,0,255), width = 2)

    im.save(save_path)


def build_line(index, line_num, line_antipod_x, line_anchor_y, line_antipod_y, word_df, separator_df,spaces):
    """
    Recursively fill the df with the line number of each word.
    
    index : the index of the current word
    line_num : the current line number
    word_df : the dataframe to update (done inplace)
    
    max_deviation_space : max_relative_space = 2 or (median space)*(1+ max_deviation_space)
    max_relative_space : max_relative_space*word height = max distance to next word
    ray_border_cropping : border of detection zone for next word = y-boundary of line - ray_border_cropping*height
    
    
    En commençant par le mot le plus à gauche, on tente de construire une ligne. On définit une zone de détection
    dans un rayon partant du dernier mot et délimité par y_intersept_min, y_intersept_max et x_anchor_max.
    Les y sont déterminés par la hauteur actuelle de la ligne à laquelle on soustrait une portion de celle-ci,
    par exemple, y parcours de +1/4 à +3/4 de la hauteur de la ligne.
    x_anchor_max est déterminé par la hauteur de la ligne (corrélée à la taille de police)
    multiplée par un coefficient. Ce coefficient est fixé par défaut à 2.
    Quand la ligne a assez avancé, il s'adapte aux espaces entre les mots précédents.
    Si la box suivante est un séparateur ou s'il n'y en a aucune, la ligne se termine.
    """
    
    
    
    max_deviation_space = 3/2
    max_relative_space = 2 if len(spaces) < 5 else np.median(spaces)*(1+max_deviation_space)
    ray_border_cropping = 1/4
    
    #Update the line number of the current word
    word_df.loc[index, "line"] = line_num
    
    current_word = word_df.loc[index]
    
    line_anchor_y = min(current_word["anchor_y"],line_anchor_y)
    line_antipod_y = max(current_word["antipod_y"],line_antipod_y)
    line_antipod_x = max(current_word["antipod_x"],line_antipod_x)
    
    line_height = line_antipod_y - line_anchor_y
    
    
    # Define the valid zone for the next word to be and select words within this zone
    y_intersept_min = line_anchor_y + ray_border_cropping*line_height
    y_intersept_max = line_antipod_y - ray_border_cropping*line_height
    x_anchor_min = line_antipod_x
    x_anchor_max = line_antipod_x + max_relative_space*line_height
    
    
    
    
    
    # Explication : Pour résoudre les problèmes de chevauchement,
    # on permet aussi aux mots dans la zone de la ligne mais derrière le front actuel
    # d'être intégrés à la ligne.
    # x_anchor_min n'est dès lors réservé qu'aux séparateurs.
    
    ###
    def common_zone_cond(df):
        return (df["antipod_y"] > y_intersept_min) & (df["anchor_y"] < y_intersept_max) & \
                 (df["anchor_x"] < x_anchor_max)
    ###
    
    word_cond = word_df["line"].isnull()
    
    # We check for antipod_x, not anchor_x, because we also need to catch "word" that
    # may be detected inside separators (for exemple in complex separators with decorations)
    separator_cond = separator_df["antipod_x"] > x_anchor_min
    
    next_word_candidates = word_df[common_zone_cond(word_df) & word_cond]
    separator_candidates = separator_df[common_zone_cond(separator_df) & separator_cond]
    
    
    #Eliminate words that are far smaller or far bigger than current word
    # 3 was too restrictive because some words can be 3 time smaller that words of the same font.
    max_height_ratio = 4
    
    size_cond = (next_word_candidates["height"] < max_height_ratio*line_height) & \
                (next_word_candidates["height"]*max_height_ratio > line_height)
    next_word_candidates = next_word_candidates[size_cond]
    
    
    # Find the next word
    if not next_word_candidates.empty:
        
        # Check if the next word is a separator (may also happens within)
        
        met_separator = False
        if not separator_candidates.empty:
            # The second line is for special cases where the current "word" is inside a separator.
            if (separator_candidates["anchor_x"].min() < next_word_candidates["anchor_x"].min()) or \
                (separator_candidates["antipod_x"].min() < next_word_candidates["anchor_x"].min()):
                met_separator = True
        
        if not met_separator:
            
            next_word_index = next_word_candidates["anchor_x"].idxmin()
            
            next_word_anchor_x = next_word_candidates["anchor_x"].min()
            
            if next_word_anchor_x > line_antipod_x:
                spaces.append((next_word_anchor_x - line_antipod_x)/line_height)
            
            build_line(next_word_index, line_num, line_antipod_x, line_anchor_y, line_antipod_y,
                       word_df, separator_df,spaces)
            
            
def segment_image(bw_im_path, im_path = None, verbose_draw = False, draw_path = None,return_invalid_box_df=False):
    """
    Do the whole segmentation process from a black and white image path and return the segments as a list of boxes.
    bw_im_path : The path of the binarized tif image
    im_path : The path of the original jpg image. Used only if verbose_draw is true
    verbose_draw : If True, the boxes are drawn on the image
    draw_path : file path where to save the drawn image. Do not need to be specified if verbose_draw is False
    return_invalid_box_df : If True, the function also return the dataFrame of invalid boxes
    (needed for the reordering function)
    Return : A list of 4-tuples. If return_invalid_box is True, return a list of 4-tuple and a dataFrame
    """
    
    # Generation of the initial connected component boxes
    im = Image.open(bw_im_path)
    box_stats = generate_all_ConnectedComponents_stats(im)
    invalid_box_df = generate_invalid_box_stats_df(box_stats,im)
    valid_box_df = generate_wordy_box_stats_df(box_stats,im)
    
    # Creating other fundamental dataFrames
    word_df = aggregate_to_next_level(valid_box_df,"word")
    separator_box_df = invalid_box_df[invalid_box_df.apply(lambda r : is_black_separator(r,im),axis = 1)]
    
    if verbose_draw:
        draw_boxes(im_path,separator_box_df,
                   draw_path+IV_util.get_cote(im_path)+"_"+IV_util.get_page(im_path,".jpg")+"_separators.jpg")
        
        draw_boxes(im_path,word_df,
                   draw_path+IV_util.get_cote(im_path)+"_"+IV_util.get_page(im_path,".jpg")+"_words.jpg")
        
    # Building the lines
    word_df["line"] = np.nan
    line_num = 0
    while word_df["line"].isnull().any():
        #print(word_df["line"].isnull().sum(),end = " ")
        # Scanning left to right : select the next word that has no line
        first_word_index = word_df[word_df["line"].isnull()]["anchor_x"].idxmin()

        first_word = word_df.loc[first_word_index]
        line_antipod_x = first_word["antipod_x"]
        line_anchor_y = first_word["anchor_y"]
        line_antipod_y = first_word["antipod_y"]

        build_line(first_word_index,line_num, line_antipod_x, line_anchor_y, line_antipod_y,
                   word_df, separator_box_df, [])

        line_num += 1
    
    # Further formatting
    word_df["line"] = word_df["line"].astype(int)
    line_df = aggregate_to_next_level(word_df, "line")
    
    line_df["words_len"] = line_df.reset_index()["line"].apply(lambda i : list(word_df[word_df["line"] == i]["length"]))
    
    # Valid lines filtering
    not_singleton_cond = line_df["words_len"].apply(lambda l : l != [1]) # Filter line of one word of one character
    horizontal_line_cond = line_df["height"] < line_df["width"]
    line_df["valid"] = not_singleton_cond & horizontal_line_cond
    
    if verbose_draw:
        draw_boxes(im_path, line_df,
                   draw_path+IV_util.get_cote(im_path)+"_"+IV_util.get_page(im_path,".jpg")+"_lines.jpg")
        
    # Final formatting
    final_df = line_df[line_df.valid]
    final_lines = final_df.apply(lambda r : (r["anchor_x"],r["anchor_y"],r["antipod_x"],r["antipod_y"]),axis = 1).to_list()
    
    if not return_invalid_box_df:
        return final_lines
    else:
        return final_lines, invalid_box_df
    
    
#### Line reordering

class Counter():
    """
    This class define a small custom counter that can be incremented with Counter += 1
    """
    
    def __init__(self):
        self.i = 0
    
    def __iadd__(self,other):
        " Overload += "
        self.i += other
        return self
        
    def get(self):
        return self.i
    
    
def find_next_non_treated_full_line(df,lower_limit_margin_ratio):
    """
    Find the next full line of text lines that have not been treated.
    df : the lines df. Columns must include anchor_x,anchor_y,antipod_x,antipod_y,<col>.
    Return : a dataFrame
    
    A full line is defined as the set of all text line that are on the same geometrical line (they share y values).
    A line is not treated iff its value for the column col is null.
    """
    
    non_treated_df = df[pd.isnull(df["num"])]
    
    if len(non_treated_df) > 0:
    
        highest_line_index = non_treated_df["anchor_y"].idxmin()

        highest_line = df.loc[highest_line_index]
        all_highest_lines = non_treated_df[non_treated_df["anchor_y"] <= highest_line["antipod_y"] - 
                                          lower_limit_margin_ratio*(highest_line["antipod_y"]-highest_line["anchor_y"])]

        return all_highest_lines


def find_down_neighbors(col_lower_limit,col_right_limit,df):
    """
    Find in the df the next full line below the current row.
    This next full line is defined by all the lines that are under the current row,
    and whose beginning starts before the current row ends (in the x direction)
    
    col_lower_limit : the lower limit of the current column. Defined as the bottom of the previous word
    col_right_limit : the right limit of the current column. Defined as the largest line met in the current column
    Return : a dataFrame containing the down neighbors, if any
    """
    
    candidates = df[(df["anchor_y"] > col_lower_limit) & (df["anchor_x"] < col_right_limit)]
    # Note that we also accept the candidate that are fully to the left of the current row.
    # This is an expected behavior that allow us to handle corner cases.
    
    if not candidates.empty:
        head_of_neighbor_line_index = candidates["anchor_y"].idxmin()
        head_of_neighbor_line = candidates.loc[head_of_neighbor_line_index]

        all_of_next_line = candidates[candidates["anchor_y"] < head_of_neighbor_line["antipod_y"]]
        
        return all_of_next_line
    
    else:
        return pd.DataFrame([])


def create_column(head_line,previous_col_mask,full_df,horizontal_seps, lower_limit_margin_ratio):
    """
    Create a mask for the column that starts with next_line.
    head_line : a dataFrame row that represent the starting line of the column.
    previous_col_mask : the mask representing all the line that belong to the super-column we are working in.
    full_df : the dataFrame containing all lines.
    horizontal_seps : the dataFrame containing all horizontal black separators
    
    The column that starts with next_line is defined as all the lines already inside previous_col_mask
    that are under next_line and to the left of the column right boundary that is dynamically defined.
    The algorithm is greedy and add lines that are lower and lower until it find a full line
    that break the right boundary or meet a wide horizontal separator.
    
    We define an horizontal separator capable of breaking a column as a separator that is larger than
    the current column.
    
    We define the outer boundary of a column as the beginning of the next column.
    We define the inner boundary of a column as the end of this column.
    """
    
    
    # We consider only the line of the supercolumn we are working in
    valid_df = full_df[previous_col_mask]
    pd.set_option('mode.chained_assignment', None)
    valid_df["is_current_col"] = False
    pd.set_option('mode.chained_assignment', "warn")
    
    
    # Loop initialisation
    
    valid_df.loc[head_line.name,"is_current_col"] = True
    
    current_line = head_line
    right_outer_boundary = valid_df["antipod_x"].max() # Init the outer boundary as the most extrem value possible.
    # This should change on the first iteration iff there is a neighboring right column.
    
    right_inner_boundary = current_line["antipod_x"]
    
    
    while current_line is not None:

        # Updating the right boundary of the current column
        
        right_neighbors = valid_df[(valid_df["anchor_y"] < current_line["antipod_y"]) &
                                   (valid_df["antipod_y"] > current_line["anchor_y"]) &
                                   (valid_df["anchor_x"] > current_line["antipod_x"])]
        
        if len(right_neighbors) > 0:
            right_outer_boundary = min(right_outer_boundary,right_neighbors["anchor_x"].min())
        
        right_inner_boundary = max(right_inner_boundary,current_line["antipod_x"])
                
        # Finding the down neighbors and updating the informations according to their position
        # The lower boundary is not the antipod_y, because lines can overlap.
        # Lines can overlap because the ray that construct the line is not as large as the line height.
        lower_boundary = current_line["antipod_y"]-lower_limit_margin_ratio*(
            current_line["antipod_y"]-current_line["anchor_y"])
        
        down_neighbors = find_down_neighbors(lower_boundary,right_inner_boundary,valid_df)
        
        if len(down_neighbors) > 0:
            down_neighbors_antipod_x = down_neighbors["antipod_x"].max()
            down_neighbors_anchor_y = down_neighbors["anchor_y"].min()
            column_width = right_outer_boundary - valid_df["anchor_x"].min() # The column is always the leftmost
            
            if not horizontal_seps.empty:
                valid_separators = horizontal_seps[(horizontal_seps["anchor_y"] < current_line["antipod_y"]) &
                                              (horizontal_seps["antipod_y"] > down_neighbors_anchor_y) &
                                              (horizontal_seps["width"] > column_width)]
            else:
                valid_separators = pd.DataFrame([])
            
            # If the column must continue, all lines of down full line are included in the column
            # and the rightmost of these lines become the next current line.
            if down_neighbors_antipod_x < right_outer_boundary and valid_separators.empty:
                pd.set_option('mode.chained_assignment', None)
                valid_df.loc[down_neighbors.index,"is_current_col"] = True
                pd.set_option('mode.chained_assignment', "warn")
                
                index_rightmost_neighbor = down_neighbors["antipod_x"].idxmax()
                current_line = valid_df.loc[index_rightmost_neighbor]
                
            else:
                current_line = None
        
        else:
            current_line = None
    
    new_column_df = valid_df[valid_df["is_current_col"]]
    return full_df.index.isin(new_column_df.index)


def process_column(col_mask,counter,full_df,horizontal_seps):
    """
    Give numbers to elements of a column. Recursively create subcolumns if needed.
    col_mask : a mask for the full df indicating only the lines of the current column.
    counter : the global counter used to number the lines.
    full_df : the dataFrame containing all lines.
    horizontal_seps : the dataFrame containing all horizontal black separators
    
    We use a mask for the columns because we need to modify always the full df,
    so that the changes are always visible for all level of recursion.
    
    Un jour, cette fonction pourrait être améliorée pour merge toutes les lignes (délimitées par une box)
    en une seule si elle traite une "colonne" de une seule ligne géométrique (full line).
    En effet, cette colonne ne serait alors qu'une ligne unique découpée en morceaux, comme cela arrive parfois.
    """
    lower_limit_margin_ratio = 1/4
    
    highest_full_line = find_next_non_treated_full_line(full_df[col_mask],lower_limit_margin_ratio)
    
    while highest_full_line is not None: # Else, every line have been treated.
    
        if len(highest_full_line) == 1:
            index_to_be_numbered = highest_full_line.iloc[0].name
            full_df.loc[index_to_be_numbered,"num"] = counter.get()
            counter += 1

        elif len(highest_full_line) > 1:
            next_index = highest_full_line["anchor_x"].idxmin()
            next_line = highest_full_line.loc[next_index]
            
            next_level_mask = create_column(next_line,col_mask,full_df,horizontal_seps,lower_limit_margin_ratio)
            process_column(next_level_mask,counter,full_df,horizontal_seps)
        
        highest_full_line = find_next_non_treated_full_line(full_df[col_mask],lower_limit_margin_ratio)
        
        
def create_segment_file(bw_im_path, save_path):
    """
    Perform the full segmentation process from the binarized image to the segments json file
    bw_im_path : the path of the black-and-white image to process
    save_path : the path of the json file to be created
    """
    
    
    # Initialisation : creation of the needed dataFrames using segment_image function
    im = Image.open(bw_im_path)
    
    lines, invalid_box_stats = segment_image(bw_im_path,return_invalid_box_df = True)
    
    width_height_max_ratio = 4
    horizontal_separators = invalid_box_stats[
        invalid_box_stats.width > invalid_box_stats.height*width_height_max_ratio]
    
    lines_df = pd.DataFrame(lines, columns = ["anchor_x","anchor_y","antipod_x","antipod_y"])
    lines_df["num"] = None
    
    # Reordering the lines
    tautology_mask = (lines_df.index == lines_df.index)
    counter = Counter()
    process_column(tautology_mask,counter,lines_df,horizontal_separators)
    
    # Creating the file
    segments = list(
        lines_df.sort_values("num")[["anchor_x","anchor_y","antipod_x","antipod_y"]].values.tolist())
    
    json_text = """{"text_direction": "horizontal-lr", "boxes": """+str(segments)+""", "script_detection": false}"""
    
    with open(save_path,"w") as f:
        f.write(json_text)