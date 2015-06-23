import vigra
import numpy as np
import copy
from graphical_models import run_mc_opengm

class my_graph(object):

    def __init__(self, segmentation, raw_data):

        self.segmentation = segmentation
        self.raw_data     = raw_data
        self.labels       = np.unique( segmentation ) - 1
        self.N_segments   = self.labels.shape[0]

        self.ignore_label      = 0
        self.unconnected_label = - 1

        self.edges        = self.get_edges()
        self.segment_feats, self.segfeat_dict = self.get_segment_features()
        self.edge_feats, self.edgefeat_dict  = self.get_edge_features()


    def get_edges(self):

        edges = []

        for z in range(self.segmentation.shape[2]-1):

            segment_ids = np.unique( self.segmentation[:,:,z] )

            #TODO make this clean
            if segment_ids[0] == self.ignore_label:
                segment_ids  = np.delete( segment_ids, 0 )
            i = segment_ids.min()

            for id in segment_ids:
                if i != id:
                    raise RuntimeError("Nonconsecutive ids for" + str(id) + " " + str(i))
                i += 1

                # look for connected segments in the next slice
                pix = np.where(self.segmentation[:,:,z] == id)
                assert len(pix) == 2
                above = self.segmentation[:,:,z+1][pix]
                conn_ids = np.unique(above).astype(np.uint32)
                #TODO make this clean
                if conn_ids[0] == self.ignore_label:
                    conn_ids = np.delete(conn_ids, 0)
                if conn_ids.size == 0:
                    continue
                for conn_id in conn_ids:
                    edges.append( (id, conn_id) )

        return edges


    def get_segment_features(self):

        regfeat_acc = vigra.analysis.extractRegionFeatures( vigra.Volume(self.raw_data),
                                                            vigra.Volume(self.segmentation).astype(np.uint32),
                                                            ignoreLabel = self.ignore_label )

        feature_selection = ["Histogram",
                            "Count",
                             "Kurtosis",
                             "Maximum",
                             "Minimum",
                             "Quantiles",
                             "RegionRadii",
                             "Skewness",
                             "Sum",
                             "Variance"
                         ]

        feature_dict = { "Cms" : 0 }
        for i in range( len(feature_selection) ):
            feature_dict[feature_selection[i]] = i

        segment_feats = { 0 : [0.] }

        for z in range(self.segmentation.shape[2]):

            segment_ids  = np.unique(self.segmentation[:,:,z]).astype(np.uint32)
            if segment_ids[0] == self.ignore_label:
                segment_ids  = np.delete( segment_ids, 0 )

            for id in segment_ids:

                features = []

                # calculate the cms for this segment
                pix = np.where( self.segmentation[:,:,z] == id )
                x_coords = np.array(pix[0])
                y_coords = np.array(pix[1])
                N_pix    = x_coords.shape[0]
                assert N_pix == y_coords.shape[0]
                cms_x    = round( x_coords.sum(axis=0) / N_pix )
                cms_y    = round( y_coords.sum(axis=0) / N_pix )
                cms_z    = z
                cms = np.array( [ cms_x, cms_y, cms_z ] )

                features.append( cms )

                for f in feature_selection:
                    features.append(regfeat_acc[f][id])

                segment_feats[id] = features

        return segment_feats, feature_dict


    def get_edge_features(self):

        edge_features = dict()

        for i in range( len(self.edges) ):
            edge = self.edges[i]
            edge_feats  = np.zeros( len(self.segfeat_dict.keys()) + 1 )

            # calculate the overlap
            pix_dn      = np.where( self.segmentation == edge[0] )
            pix_up      = np.where( self.segmentation == edge[1] )
            pix_overlap = np.where(
                    np.logical_and(pix_dn[0] == pix_up[0], pix_dn[1] == pix_up[1]) == True )

            overlap = pix_overlap[0].size

            edge_feats[-1] = overlap

            feats_dn = self.segment_feats[ edge[0] ]
            feats_up = self.segment_feats[ edge[1] ]

            cms_dn = feats_dn[ self.segfeat_dict["Cms"] ]
            cms_up = feats_up[ self.segfeat_dict["Cms"] ]

            assert cms_dn[2] == cms_up[2] - 1

            cms_dist = ( cms_dn[0] - cms_up[0] )**2 + ( cms_dn[1] - cms_up[1] )**2

            edge_feats[self.segfeat_dict["Cms"]] = cms_dist

            for feat in self.segfeat_dict.keys():
                if feat == "Cms":
                    continue
                feat_index = self.segfeat_dict[feat]
                if feats_dn[feat_index].size == 1:
                    edge_feats[feat_index] = np.abs( feats_dn[feat_index] - feats_up[feat_index] )
                else:
                    summ = 0.
                    for f in range(feats_dn[feat_index].size):
                        summ += np.abs( feats_dn[feat_index][f] - feats_up[feat_index][f] )
                    edge_feats[feat_index] = summ

            edge_features[i] = edge_feats

        edgefeat_dict = copy.deepcopy( self.segfeat_dict )
        edgefeat_dict["Overlap"] = len(self.segfeat_dict.keys() )

        return edge_features, edgefeat_dict

    def get_edge_energies(self):

        edge_probs = np.zeros( len(self.edges) )
        edge_energies = np.zeros( len(self.edges) )

        for i in range( len(self.edges) ):
            edge_feat = self.edge_feats[i]
            ovlp = edge_feat[self.edgefeat_dict["Overlap"]]
            edge_probs = np.exp(-ovlp) / ( 1. + np.exp(-ovlp) )
            edge_energies = ovlp

        return edge_energies

    def get_connectivity(self, edge_energies):

        run_mc_opengm(self.segmentation, self.edges, edge_energies)
        #run_mc_ccc3d(self.segmentation, self.edges, self.energies)


    def get_neurons(self, connectivity):
        neurons = { }
        start_seg_max = np.unique(self.segmentation[:,:,0]).max()
        start_already_visited = []
        for start_seg in np.unique(self.segmentation[:,:,0]):
            if start_seg == self.ignore_label:
                continue
            if start_seg in start_already_visited:
                continue
            neurons[start_seg] = []
            visit = [start_seg]
            while visit:
                i = visit.pop()
                for child in connectivity[i]:
                    if not child in neurons[start_seg]:
                        visit.append(child)
                        neurons[start_seg].append(child)
                        if child <= start_seg_max:
                            start_already_visited.append(child)
        return neurons


    def nrns_to_segmentation(self, neurons):
        new_seg = np.zeros( self.segmentation.shape )
        assigned_to_nrn = []
        for n_id in neurons.keys():
            for id in neurons[n_id]:
                if id in assigned_to_nrn:
                    raise RuntimeError("Segment assigned to two different neuron ids: " + str(id) + " " + str(n_id))
                new_seg[self.segmentation==id] = n_id
                assigned_to_nrn.append(id)
        return new_seg
