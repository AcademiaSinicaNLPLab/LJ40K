# -*- coding: utf-8 -*-
"""

"""
from PIL import Image, ImageDraw
import os, sys
import pymongo
import logging
from collections import defaultdict

class ImageUtil(object):
    """
    from feelit.image import ImageUtil

    """
    def __init__(self):
        NUMERALS = '0123456789abcdefABCDEF'
        self._HEXDEC = {v: int(v, 16) for v in (x+y for x in NUMERALS for y in NUMERALS)}
        self._LOWERCASE, self._UPPERCASE = 'x', 'X'

    def triplet(self, rgb, **kwargs):
        """
        Transform RGB to HEX

        triplet( rgb=(255,255,255) ) will output ffffff

        Parameters
        ==========

        rgb: tuple (3 elements)
            triple of R, G and B (0-255)

        options:
            lower: True/False
            upper: True/False

        Returns
        =======

        6-digit HEX code (e.g., `FFFFFF`)
        """
        if 'lower' in kwargs and kwargs['lower'] == True or 'upper' in kwargs and kwargs['upper'] == False: lettercase = self._LOWERCASE
        if 'upper' in kwargs and kwargs['upper'] == True or 'lower' in kwargs and kwargs['lower'] == False: lettercase = self._UPPERCASE
        else: lettercase = self._LOWERCASE

        return format(rgb[0]<<16 | rgb[1]<<8 | rgb[2], '06'+lettercase)

    def rgb(self, triplet):
        try:
            rbg_code = self._HEXDEC[triplet[0:2]], self._HEXDEC[triplet[2:4]], self._HEXDEC[triplet[4:6]]
            return rbg_code
        except KeyError:
            return False

    def RGBA_to_RGB(self, rgba, bg=(255,255,255)):
        ''' 
        convert RGBA to RGB upder different background
        '''
        bg_r, bg_g, bg_b = bg
        if type(rgba) not in (tuple, list) or len(rgba) != 4:
            raise TypeError('rgba is a tuple or list containing four values')

        r, g, b, a = rgba
        return tuple(map(lambda x:int(x), [(1-a)*bg_r + a*r, (1-a)*bg_g + a*g, (1-a)*bg_b + a*b]))        


class ImageDrawer(object):
    """
    Draw images containing emotion of each pattern in LJ40K

    Parameters
    ==========

    color_order: tuple
        default: ('feelit', 'color.order')
    color_map: tuple
        default: ('feelit', 'color.map')
    color_theme: str
        default: 'default'

    Usage
    =====
    from feelit.image import ImageDrawer

    """
    def __init__(self, **kwargs):
        
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        ### mongodb settings
        mongo_addr = 'doraemon.iis.sinica.edu.tw' if 'mongo_addr' not in kwargs else kwargs['mongo_addr']

        ## default db and collection names for color order and map
        color_order = ('feelit', 'color.order') if 'color_order' not in kwargs else kwargs['color_order']
        color_map = ('feelit', 'color.map') if 'color_map' not in kwargs else kwargs['color_map']
        color_theme = 'default' if 'color_theme' not in kwargs else kwargs['color_theme']

        ## default collection name
        lexicon = 'lexicon.nested' if 'lexicon' not in kwargs else kwargs['lexicon']
        pats = 'pats' if 'pats' not in kwargs else kwargs['pats']
        patscore = 'patscore.normal' if 'patscore' not in kwargs else kwargs['patscore']
        self._db = 'LJ40K' if 'db' not in kwargs else kwargs['db']


        ### connect to mongodb
        self._mc = pymongo.MongoClient(mongo_addr)

        self._co_color_order = self._mc[color_order[0]][color_order[1]]
        self._co_color_map = self._mc[color_map[0]][color_map[1]]

        self._co_lexicon = self._mc[self._db][lexicon]
        self._co_pats = self._mc[self._db][pats]
        self._co_patscore = self._mc[self._db][patscore]

        ### shared 
        self.emo_list = self._co_color_order.find_one({ 'order': 'group-maxis'})['emotion']

        ## get theme color mapping
        self.color_mapping = self._co_color_map.find_one({'theme': color_theme})['map']

        self._white = (255,255,255)

    def accumulate_threshold(self, dist, percentage):
        """
        Cut the distribution based on the accumulated threshold
        input: distirbution
        output: pruned distirbution
        by @Chenyi-Lee
        """
        ## temp_dict -> { 0.3: ['happy', 'angry'], 0.8: ['sleepy'], ... }
        ## (dist)      { 2:   ['bouncy', 'sleepy', 'hungry', 'creative'], 3: ['cheerful']}
        temp_dict = defaultdict( list ) 
        for e in dist:
            temp_dict[dist[e]].append(e)
        
        ## temp_list -> [ (0.8, ['sleepy']), (0.3, ['happy', 'angry']), ... ] ((sorted))
        ## (dist)      [ (3, ['cheerful']), (2,   ['bouncy', 'sleepy', 'hungry', 'creative'])]
        temp_list = temp_dict.items()
        temp_list.sort(reverse=True)

        th = percentage * sum( dist.values() )
        current_sum = 0
        selected_emotions = []

        while current_sum < th:
            top = temp_list.pop(0)
            selected_emotions.extend( top[1] )
            current_sum += top[0] * len(top[1])

        return dict( zip(selected_emotions, [1]*len(selected_emotions)) )

    def listPatterns(self, udocID):
        """
        Fetch patterns in the document with given udocID
        input: udocID
        output: self.Sent2Pats
        """
        ## fetch
        mdocs = self._co_pats.find({'udocID': udocID}, {'_id':0, 'pattern':1, 'usentID': 1, 'weight':1}).batch_size(512)
        ## group by sent
        self.Sent2Pats = defaultdict(list)
        for mdoc in mdocs:
            self.Sent2Pats[mdoc['usentID']].append( (mdoc['pattern'], mdoc['weight']) )
        return self.Sent2Pats

    def getPatDists(self, **kwargs):
        """
        Get pattern distributions
        input: self.Sent2Pats or Sent2Pats (to overwrite self.Sent2Pats)
        options: 
            scoring:    *False/True
            weighted:   *False/True
            min_count:  *1 <positive integer number>
        output: self.dists
        """
        ## default value
        scoring = False if 'scoring' not in kwargs else kwargs['scoring']
        weighted = False if 'weighted' not in kwargs else kwargs['weighted']
        min_count = 1 if 'min_count' not in kwargs else int(kwargs['min_count'])

        ## overwrite Sent2Pats
        if 'Sent2Pats' in kwargs: 
            self.Sent2Pats = kwargs['Sent2Pats']

        self.dists = defaultdict(list)

        for usentID in self.Sent2Pats:

            for pat, weight in self.Sent2Pats[usentID]:

                pat = pat.lower()

                mdoc = self._co_lexicon.find_one({'pattern': pat}) if not scoring else self.co_patscore.find_one({'pattern': pat})

                if not mdoc or sum(mdoc['count'].values()) <= min_count:
                    continue
                else:
                    pass

                dist = mdoc['count'] if not scoring else mdoc['score'] 

                w = weight if weighted else 1.0

                vector = {}
                for e in self.emo_list:
                    if e not in dist: dist[e] = 0.0
                    vector[e] = dist[e]*w

                self.dists[usentID].append( vector )
        return self.dists

    def aggregate(self, **kwargs):
        """
        Aggregate based on sentence or pattern
        and update self.dists

        input: self.dists or dists (to overwrite dists)
        output: self.dists (aggregated distribution)
        """

        ## default value
        base = 'pattern' if 'base' not in kwargs else kwargs['base']

        ## overwrite dists
        if 'dists' in kwargs: 
            self.dists = kwargs['dists']

        ## aggregate based on sentence or pattern
        vectors = []
        if self.dists.values():
            ## sentence base
            if base.startswith('sent'):
                for usentID in self.dists:
                    aggregated_vector = {}
                    for vector in self.dists[usentID]:
                        if aggregated_vector:
                            for e in aggregated_vector: aggregated_vector[e] += vector[e]
                        else:
                            aggregated_vector = dict(vector)
                    vectors.append( aggregated_vector )
            ## pattern base
            else:
                vectors = reduce(lambda x,y:x+y, self.dists.values())

        self.dists = vectors

        return self.dists

    def generateColorMatrix(self, **kwargs):

        ## overwrite dists
        if 'dists' in kwargs: 
            self.dists = kwargs['dists']

        alpha = True if 'alpha' not in kwargs else kwargs['alpha']
        percent = 1.0 if (alpha == True) or ('percent' not in kwargs) else kwargs['percent']

        IU = ImageUtil()

        ## dists, percent, alpha=True
        self.matrix = []
        for dist in self.dists:

            ## clear render color
            render_colors = { e: self._white for e in self.emo_list }

            E = self.accumulate_threshold(dist, percent)

            filtered = { e: dist[e] for e in E }
            S = float(sum(filtered.values()))
            filtered_normalized = { e: filtered[e]/S for e in filtered }

            for e in filtered_normalized: 
                if not alpha:
                    rgb = tuple(self.color_mapping[e]['rgb'])
                else:
                    alpha_value = filtered_normalized[e]
                    rgba = tuple( list(self.color_mapping[e]['rgb']) + [alpha_value] )
                    rgb = IU.RGBA_to_RGB(rgba, bg=self._white)
                render_colors[e] = rgb

            order_render_colors = [(e,render_colors[e]) for e in self.emo_list]
            
            row = []
            for emotion, color in order_render_colors:
                row.append(color)
            self.matrix.append(row)

        return True if self.matrix else False

    def draw(self, **kwargs):
        #matrix, output_path, w=5, h=5, rows='auto'

        ## overwrite matrix
        if 'matrix' in kwargs: 
            self.matrix = kwargs['matrix']

        ## default values
        w = 1 if 'w' not in kwargs else kwargs['w']
        h = 1 if 'h' not in kwargs else kwargs['h']

        img_width = w*len(self.matrix[0])
        img_height = h*len(self.matrix)

        self.image = Image.new('RGB', (img_width, img_height))

        ## draw a rectangle
        draw = ImageDraw.Draw(self.image)

        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[row])):

                x0, x1 = col*h, (col+1)*h
                y0, y1 = row*w, (row+1)*w
                
                fillcolor = self.matrix[row][col]
                draw.rectangle(xy=[(x0,y0),(x1,y1)], fill=fillcolor)
    def draw_blank(self, **kwargs):
        ## default values
        w = 1 if 'w' not in kwargs else kwargs['w']
        h = 1 if 'h' not in kwargs else kwargs['h']

        img_width = w
        img_height = h

        self.image = Image.new('RGB', (img_width, img_height))
        
        draw = ImageDraw.Draw(self.image)
        draw.rectangle(xy=[(0,0),(w,h)], fill=self._white)

    ## path-related functions
    def generatePaths(self, parent, docs, w, h, alpha, base):

        self._rootFolder = {}
        for udocID, emotion in docs:
            postfix = 'rgb' if not alpha else 'rgba'
            output_folder = '/'.join([parent, 'emotion-imgs-%dx%d-%s' % (w, h, postfix), base, emotion])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            self._rootFolder[udocID] = output_folder

    def getRoot(self, udocID):
        return self._rootFolder[udocID]

    def getFileName(self, udocID, alpha, ext='png'):
        postfix = 'rgba' if alpha else 'rgb'
        return '.'.join([str(udocID), postfix, 'png'])

    def save(self, fname="default.png", root="", overwrite=False):
        if os.path.isfile(fname) and not overwrite:
            logging.error('output file %s already exists, set overwrite=True to overwrite it' % (fname))
            return False
        else:
            self.image.save( os.path.join(root, fname) )
            return True

    def batchRun(self, **kwargs):

        ## getPatDists
        scoring = False if 'scoring' not in kwargs else kwargs['scoring']
        weighted = False if 'weighted' not in kwargs else kwargs['weighted']
        min_count = 1 if 'min_count' not in kwargs else int(kwargs['min_count']) 
        
        ## aggregate
        base = 'pattern' if 'base' not in kwargs else kwargs['base']

        ## generateColorMatrix
        alpha = True if 'alpha' not in kwargs else kwargs['alpha']
        percent = 1.0 if (alpha == True) or ('percent' not in kwargs) else kwargs['percent']

        root_path = '.' if 'root_path' not in kwargs else kwargs['root_path']

        ## draw
        w = 1 if 'w' not in kwargs else kwargs['w']
        h = 1 if 'h' not in kwargs else kwargs['h']

        ## generateFileName
        ext = 'png' if 'ext' not in kwargs else kwargs['ext']

        ## get documents
        logging.info('loading udocIDs and emotions')
        docs = sorted([ (x['udocID'], x['emotion']) for x in self._mc[self._db]['docs'].find().batch_size(1024)], key=lambda x:x[0] )
import pdb; pdb.set_trace()
        logging.info('generating destination folders under %s' % (root_path))
        self.generatePaths(root_path, docs, w, h, alpha, base)

        logging.info('forming pattern emotion images')
        for udocID, emotion in docs:

            logging.info('processing %d' % (udocID))

            logging.debug('fetch patterns of %d' % (udocID))

            self.listPatterns(udocID)
            self.getPatDists(scoring=scoring, weighted=weighted, min_count=min_count)
            self.aggregate(base=base)

            isValidMatrix = self.generateColorMatrix(alpha=alpha, percent=percent)

            if isValidMatrix:
                ## draw image
                logging.debug('draw patterns image %d' % (udocID))
                self.draw(w=w, h=h)
            else:
                ## draw blank image
                self.draw_blank(w=w, h=h)
                logging.debug('no image for %d; create a blank one' % (udocID))

            ## save image
            fn = self.getFileName(udocID, alpha, ext=ext)
            root = self.getRoot(udocID)
            logging.debug('save image of %d under %s' % (udocID, root))
            self.save(fname=fn, root=root)

if __name__ == '__main__':

    # from feelit.image import ImageDrawer

    ID = ImageDrawer(db='LJ40K', verbose=False, color_theme='black')

    # ID.listPatterns(38800)
    # ID.getPatDists()
    # ID.aggregate()
    # ID.generateColorMatrix()
    # ID.draw(w=5,h=5)
    # ID.save(fname="38800.rgba.png")

    ID.batchRun(w=1, h=1, scoring=False, weighted=False, min_count=1, base="pattern", alpha=True, root_path='images')
    # ID.batchRun(w=1, h=1, scoring=False, weighted=False, min_count=1, base="sentence", alpha=True)

    ID.batchRun(w=1, h=1, scoring=False, weighted=False, min_count=1, base="pattern", alpha=False, percent=0.5, root_path='images')
    # ID.batchRun(w=1, h=1, scoring=False, weighted=False, min_count=1, base="sentence", alpha=False, percent=0.5)

