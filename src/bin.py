"""
            print(edge_index)
            exit()
  
            for individual_index in range(n):
                pivot_value = self._total_gt.loc[individual_index, etype]

                for circular_individual_index in range(individual_index +1, n):
                    comparer_value = self._total_gt.loc[circular_individual_index, etype]

                    if pivot_value == comparer_value:
                        edge_index.append((individual_index  , circular_individual_index))


"""





"""
class HeteroGNN(nn.Module):
    def __init__(self, cfg: DictConfig):

        Initialize the Heterogeneous Graph Neural Network (HeteroGNN).

        Parameters:
        -----------
        attributes : list
            A list of attribute types used in the graph.
        
        embedding_size : int, optional
            The size of the embeddings for the attributes, by default 128.
        
        aggr : str, optional
            The aggregation method to use in HeteroConv layers, by default "mean".
        

        super().__init__()

        attribute_edges =  [("individuals", etype, etype) for etype in cfg.attributes]
        self._attributes = cfg.attributes

        ### Attributes
        self._attributes_update = nn.ModuleList()
        self._attr_ind_convs = nn.ModuleList()
        self._ind_ind_convs = nn.ModuleList()
        self._activation = nn.ReLU(inplace=False)

        
        ### ** D'aquí endavant s'ha de pensat millor

        ## ^ Inidivuals to attributes
        ind_attribute_conv = {attrribute_etype: SAGEConv((-1, -1), cfg.embedding_size, flow="source_to_target") for attrribute_etype in attribute_edges}
        
        ## ^ Inidivuals to Individuals
        ind_attribute_conv.update({
            ("individuals", "family", "individuals"): SAGEConv((-1, -1), cfg.embedding_size, flow="source_to_target"),
            ("individuals", "family", "individuals"): SAGEConv((-1, -1), cfg.embedding_size, flow="target_to_source")
        })
        ## ^ Attributes to Individuals
        ind_attribute_conv.update(
                {attrribute_etype: SAGEConv((-1, -1), cfg.embedding_size, flow="target_to_source") for attrribute_etype in attribute_edges}
        )
                
        conv = HeteroConv(ind_attribute_conv, aggr=cfg.aggr)
        self._attributes_update.append(conv)


        ### Individuals to Individuals
        conv = HeteroConv({
            ("individuals", "family", "individuals"): SAGEConv((-1, -1), cfg.embedding_size, flow="source_to_target"),
            ("individuals", "family", "individuals"): SAGEConv((-1, -1), cfg.embedding_size, flow="target_to_source")
        }, aggr=cfg.aggr)

        self._ind_ind_convs.append(conv)


    def apply_edges_on_individuals(self, x_dict:Dict[str, TensorType],
                                    population: TensorType["batch_individual_nodes"]):
            
            Aggregate attribute information back into individuals.

            Parameters:
            -----------
            x_dict : dict
                A dictionary containing the feature matrices of the nodes. 
                The keys are the node types (e.g., "individuals", attribute types), 
                and the values are the corresponding feature matrices of shape [num_nodes, num_features].

            population : list or torch.Tensor
                A list or tensor containing the indices of the individuals in the current population.

            Returns:
            --------
            x_dict : dict
                A dictionary containing the updated feature matrices of the nodes after aggregating attribute information.

        for idx, attribute in enumerate(self._attributes):
            inverse_edge_space = x_dict[attribute][population] @ (torch.inverse(self._edge_space_projection[attribute]) + self._identity)
            x_dict[attribute][population] = inverse_edge_space
            x_dict[attribute][population] = self._activation(inverse_edge_space)


        return x_dict

    
    def attribute_encoder(self, x_dict: Dict[str, TensorType],
                edge_index_dict: Dict[Tuple[str, str, str], Adj],
                edge_attributes: TensorType["batch", "Nattributes", "embedding_size"], 
                population: TensorType["batch_individual_nodes"]):
        
  
  
        for convs in self._attributes_update:
            x_dict = self.apply_edges_on_attributes(x_dict=x_dict, positional_features=edge_attributes, population=population)
            x_dict_attributes = convs(x_dict, edge_index_dict)   
            x_dict = {key: x.relu() for key, x in x_dict_attributes.items()}
            
        return x_dict   

    def forward(self, x_dict: Dict[str, TensorType],
                edge_index_dict: Dict[Tuple[str, str, str], Adj],
                edge_attributes: TensorType["batch", "Nattributes", "embedding_size"], 
                population: TensorType["batch_individual_nodes"]):


        Forward pass for the Heterogeneous Graph Neural Network (HeteroGNN).

        Parameters:
        -----------
        x_dict : dict
            A dictionary containing the feature matrices of the nodes. 
            The keys are the node types (e.g., "individuals", attribute types), 
            and the values are the corresponding feature matrices of shape [num_nodes, num_features].

        edge_index_dict : dict
            A dictionary containing the edge indices for each edge type.
            The keys are tuples representing the edge types (e.g., ("individuals", "attribute_type", "individuals")), 
            and the values are the corresponding edge index tensors of shape [2, num_edges].

        edge_attribute_dict : torch.Tensor
            A tensor containing the edge attributes/features for the edges between individuals and attributes.
            Shape: [num_edges, num_attributes, embedding_size].

        population : list or torch.Tensor
            A list or tensor containing the indices of the individuals in the current population.

        Returns:
        --------
        x_dict : dict
            A dictionary containing the updated feature matrices of the nodes after the forward pass.
            The keys are the node types (e.g., "individuals", attribute types), 
            and the values are the corresponding updated feature matrices of shape [num_nodes, num_features].

        Description:
        ------------
        The forward pass consists of three main stages:
        
        1. Applying edge attributes to the features of nodes (individuals) using the attributes:
            - For each attribute type, positional features are transformed and concatenated with node features.
            - The concatenated features are passed through a linear layer to get aggregated information.
            - This updated information is used to modify the node features of the attributes in `x_dict`.
        
        2. Aggregating attribute information back into individuals:
            - For each attribute type, the inverse of the edge space projection matrix is applied to the attribute features.
            - This inverse transformed information is used to update the individual features in `x_dict`.
        
        3. Applying convolution operations to update the node features:
            - For each convolution layer in `_ind_attribute_convs`, `_attr_ind_convs`, and `_ind_ind_convs`, 
            the respective convolutions are applied to the `x_dict` using the `edge_index_dict`.
            - After each convolution, ReLU activation is applied to the updated node features.
        
        The final updated `x_dict` contains the new node features after applying the heterogeneous graph convolutions.


        for convs in self._ind_attribute_convs:
            x_dict = self.apply_edges_on_attributes(x_dict=x_dict, positional_features=edge_attributes, population=population)
            x_dict_attributes = convs(x_dict, edge_index_dict)

        x_dict.update(x_dict_attributes)

        for convs in self._attr_ind_convs:
            x_dict = self.apply_edges_on_individuals(x_dict=x_dict, population=population)
            x_dict_individuals = convs(x_dict, edge_index_dict)

        x_dict.update(x_dict_individuals)
        

        for conv in self._ind_ind_convs:
            x_dict_family = conv(x_dict=x_dict, edge_index_dict=edge_index_dict)

        x_dict.update(x_dict_family)
 

        return x_dict

    
    
   
## ** Example 
class HGT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        
        self.lin_dict = nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author'])


"""





""""
    print(population)
    
    dirname = f"./plots/shifts/{attribute}"
    name_file = f"{attribute}_test_final.jpg"
    fig_path = os.path.join(dirname, name_file)
    
    
    x_previous = previous_x_dict[attribute].cpu().numpy()
    x_after = updated_dict[attribute].detach().cpu().numpy()
    #population = population_dict[mode] #population.cpu().numpy()
    lines = (np.array(graph_structure["image_lines"])[population])
    map_ocr = {"nom": 0, "cognom_1": 1, "cognom_2": 2}
    unique_colors = set([line._ocr[map_ocr[attribute]] for line in lines])
    map_colors = dict(zip(unique_colors, np.arange(len(unique_colors))))
    
    hue = []
    for line in lines:
        ocr = line._ocr[map_ocr[attribute]]
        color = map_colors[ocr]
        hue.append((ocr, color))

        
    utils.plot_shift(x_previous=x_previous, x_after=x_after, population=np.array(population), fig_name=fig_path, hue=hue)

    exit()

"""




"""
                tripletes = None
                for pair in edge_index_similar.T:
                    pair = pair.view(2, -1)
                    mask = ~torch.isin(edge_index_similar, pair).any(dim=0)
                    indexes_negative = edge_index_similar[:, mask].unique()
                    combinations = pair.repeat(1, indexes_negative.shape[0])
                    final = torch.cat((indexes_negative[None, :],combinations), dim=0)
                    
                    if tripletes is None:
                        tripletes = final
                        
                    else:
                        tripletes = torch.cat((tripletes, final), dim=1)
                
                anchors = x_dict[attribute][tripletes[0,:]]
                positive = x_dict[attribute][tripletes[1, :]]
                negatives = x_dict[attribute][tripletes[-1, :]]
                actual_loss = criterion(anchors, positive, negatives)

"""


"""
            for individual_index in final_nodes:
                image = self._graph["image_lines"][individual_index]._path
                image = cv.imread(image)
                image = transforms.to_tensor(image)
                _, h, w = image.shape
                if w < width:
                    
                    image = transforms.resize(img=image, size=(height , width))
                    
                else:
                    image = image[:, :, :width]
                    image = transforms.resize(img=image, size=(height , width))

                #image = image[:, :, :width//2]
                images_to_keep.append(image)

"""


"""
    def apply_edges_on_attributes(self, x_dict: Dict[str, TensorType],
                                  positional_features: TensorType["batch", "Nattributes", "embedding_size"], 
                                  population: TensorType["batch_individual_nodes"]):


        for idx, attribute in enumerate(self._attributes):

            edge_space_features_positional = (positional_features[:, idx, :] @ self._edge_space_projection[attribute]) 
            attributte_information = x_dict[attribute][population] @ self._edge_space_projection[attribute] 
            aggregation = torch.cat((attributte_information, edge_space_features_positional.cuda()), dim=1).cuda()
            aggregated_information = self._edge_space_aggregator(aggregation)

            x_dict[attribute][population] = aggregated_information


        return x_dict


"""


"""
            for volume in self._volumes:
                for page in volume._pages:
                    for line in page._individuals:

                        image_lines_path.append(line) 
                        image = T.to_tensor(line.image())
                        _, h, w = image.shape
                        
                        heights.append(h)
                        widths.append(w)
                        ar.append(w / h)

                        if w < self._width:
                            image = T.resize(img=image, size=(self._height , self._width))
                        else:
                            image = image[:, :, :self._width]
                            image = T.resize(img=image, size=(self._height , self._width))
                            
"""



"""
        true_pairs = torch.from_numpy(np.array(true_pairs)).T
        etype = ("individual", "same_as", "individual")
        self._graph[etype].edge_index = true_pairs.unique(dim=1)

        self._graph[etype].candidate_pairs = torch.cat((self._graph[("cognom_2", "similar", "cognom_2")].edge_index, true_pairs), dim=1).unique(dim=1) 
        
            
        true_pairs_undirected = torch.cat((true_pairs, true_pairs[[1, 0], :]), dim=1)
        gt_candidate_pairs = (self._graph[etype].candidate_pairs.unsqueeze(2) == true_pairs_undirected.unsqueeze(1)).all(dim=0).any(dim=1)
        
        self._graph[etype].y_candidate_pairs = gt_candidate_pairs.type(torch.int32)
        self._graph[etype].negative_sampling = self._graph[etype].candidate_pairs[:,gt_candidate_pairs.type(torch.int32)==0]  

"""

"""

    #? Define the dataset
    if  CFG_DATA.import_data == True:
        if os.path.exists(f"./pickles/graphset_{number_volum_years}_volumes_{embedding_size}_entities_{len(CFG_SETUP.configuration.compute_loss_on)}.pkl"):
            Graph = utils.read_pickle(f"./pickles/graphset_{number_volum_years}_volumes_{embedding_size}_entities_{len(CFG_SETUP.configuration.compute_loss_on)}.pkl")
    
        else:
            
            volumes = pipes.load_volumes(cfg=CFG_DATA) 
            Graph = Graphset(Volumes=volumes, auxiliar_entities_pk=pk, graph_configuration=CFG_DATA.graph_configuration)
            Graph._initialize_image_information()
            Graph._initialize_nodes(embedding_size = embedding_size)
            Graph._initialize_edges(embedding_size=embedding_size)
            
    else:
        
        volumes = pipes.load_volumes(cfg=CFG_DATA)
        Graph = Graphset(Volumes=volumes, auxiliar_entities_pk=pk, graph_configuration=CFG_DATA.graph_configuration)
        Graph._initialize_image_information()
        Graph._initialize_nodes(embedding_size = embedding_size)
        Graph._initialize_edges(embedding_size=embedding_size)
        
    if  CFG_DATA.export_data == True:

        os.makedirs("./pickles", exist_ok=True)
        utils.write_pickle(info=Graph, filepath=f"./pickles/graphset_{number_volum_years}_volumes_{embedding_size}_entities_{len(CFG_SETUP.configuration.compute_loss_on)}.pkl")
        
"""



"""
            for idx, attribute in enumerate(entities):
                
                if attribute == "individual":edge = "same_as"
                    
                else:edge = "similar"

                edge_similar_name = (attribute, edge, attribute)
                
                edge_index_similar = edge_index_dict[edge_similar_name].to(device=device)
                
                if attribute == "individual":
                    #edge_index_similar = torch.cat((edge_index_similar, edge_index_similar[[1,0],:]), dim=1)
                    all_combinations = torch.tensor(list(itertools.combinations(population.tolist(), 2)), device=edge_index_similar.device).T
                    
                    mask = ~(all_combinations.unsqueeze(2) == edge_index_similar.unsqueeze(1)).all(dim=0).any(dim=1)
                    negative_edge_index_similar = all_combinations[:, mask]
                    
                else:
                        negative_edge_index_similar = negative_edge_index_dict[edge_similar_name].to(device=device)
                
                positive_labels = torch.ones(edge_index_similar.shape[1])
                negative_labels = torch.zeros(negative_edge_index_similar.shape[1])

                gt = torch.cat((positive_labels, negative_labels), dim=0).to(device=device)
                    
                edge_index = torch.cat((edge_index_similar, negative_edge_index_similar), dim=1)
                x1 = x_dict[attribute][edge_index[0,:]]
                x2 = x_dict[attribute][edge_index[1, :]]
                visual_actual_loss = criterion(x1, x2, gt)
                
                x1_textual = x_dict[attribute][edge_index[0,:]]
                x2_textual = x_dict_ocr[attribute][edge_index[1,:]]
                mix_domain_loss1 = criterion(x1_textual, x2_textual, gt)
                
                x3_textual = x_dict_ocr[attribute][edge_index[0,:]]
                x4_textual = x_dict[attribute][edge_index[1,:]]
                mix_domain_loss2 = criterion(x3_textual, x4_textual, gt)
                

"""


"""
## Problema d'aquest, el visual encoder si que està capturant el overall de les shapes, pero aquest no
#min_height: int, max_width: int, hidden_channels:int, output_channels:int, number_of_entities:int=5, edge_embedding_size:int=128
class EdgeAttFeatureExtractor(nn.Module):
    
    def __init__(self, cfg: DictConfig) -> None:
        super(EdgeAttFeatureExtractor, self).__init__()
        
        self._height = cfg.kernel_height
        self._width = cfg.kernel_width
        self._patch_size = cfg.patch_size
        self._kernels = list(zip(self._height, self._width))
        self._input_channels = cfg.input_channels
        self._hidden_channels = cfg.hidden_channels
        self._output_channels = len(cfg.number_of_different_edges)  

        self._embedding_size = cfg.output_channels   
                                                                                    ## Amb tots els volums (10, 3), amb menys volums (5 ,3)
        self._conv1 = nn.Conv2d(in_channels=self._input_channels, out_channels=self._hidden_channels, kernel_size=self._kernels[0], padding="same")#(self._min_heigh, self._max_width))
        self._conv2 = nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._hidden_channels//2, kernel_size=self._kernels[1], padding="same")
        self._conv3 = nn.Conv2d(in_channels=self._hidden_channels//2, out_channels=self._output_channels, kernel_size=self._kernels[1], padding="same")
        
        self._add_attention = cfg.add_attention
        self._positional_max_length = cfg.positional_max_length
        self._input_attention_mechanism = cfg.input_atention_mechanism
        
        self._edge_projector_pooling = nn.MaxPool2d((2, 1)) ## image which will be with embedding 100
        
        self._attention_mechanism  = EdgeAttSelfAttention(in_dim=self._input_attention_mechanism)
        self._output_layer = nn.Linear(self._input_attention_mechanism, self._embedding_size)
        self.pos_emb = nn.Parameter(torch.randn(self._positional_max_length))
        
        self.pe = att.PositionalEncoding(d_model=self._patch_size, max_len=self._positional_max_length)

        self.attention_values = None

    def forward(self, x):

        x = self._conv1(x) 

        B, C, H, W = x.shape
        positional_repeated = self.pos_emb.repeat(*x.shape[:-1], 1)
        x = torch.relu(x) + positional_repeated[:, :, :, :x.shape[-1]]
        x = x.view(B, C, H, W)
        x = self._edge_projector_pooling(x)
        x = self._conv2(x)
        x = torch.relu(x)
        x = self._edge_projector_pooling(x)

        x = self._conv3(x)
        x = torch.relu(x)
        x = self._edge_projector_pooling(x)
        x = torch.sum(x, dim=2)
        

        if self._add_attention:
            x = x.unsqueeze(2)
            x, self.attention_values = self._attention_mechanism(x)

        x = self._output_layer(x)

        return x
    
"""






"""
    attribute_embeddings = attribute_embeddings.cpu()
    entity_embeddings = entity_embeddings.cpu()
    ## Randomly Evaluate different names and surnames:
    filtered_name_attributes = {label: indexes for label, indexes in graph["nom"].map_attribute_index.items() if len(indexes) > 20}
    filtered_surname_attributes = {label: indexes for label, indexes in graph["cognom_1"].map_attribute_index.items() if len(indexes) > 20}
    filtered_ssurname_attributes = {label: indexes for label, indexes in graph["cognom_2"].map_attribute_index.items() if len(indexes) > 20}
    
    
    
    random_name = random.sample((filtered_name_attributes.keys()), 1)[0]
    random_surname = random.sample((filtered_surname_attributes.keys()), 1)[0]
    random_second_surname = random.sample((filtered_ssurname_attributes.keys()), 1)[0]

    if save_plots:
        previous_embeddings = copy.copy(graph.x_attributes).numpy()


        ## & Name shift plot
        try:
            population = graph["nom"].map_attribute_index[random_name]
            nom_idx = list(graph.map_attribute_nodes.values()).index("nom")
            nom_embeddings = attribute_embeddings[:, nom_idx, :].numpy()


            visu.plot_shift(embedding_matrix_1=previous_embeddings[population, nom_idx,:],
                            embedding_matrix_2= nom_embeddings[population],
                            fig_path=os.path.join(base_save_plot, "nom", f"shift_{random_name}.jpg"))
        except:
            print("Visualization error in shif plot")

        visu.plot_embedding_distribution(nom_embeddings, os.path.join(base_save_plot, "nom", f"distribution.jpg"))

        ## & Surname shift plot
        try:
            population = graph["cognom_1"].map_attribute_index[random_surname]
            cognom_idx = list(graph.map_attribute_nodes.values()).index("cognom_1")
            cognom_embeddings = attribute_embeddings[:, cognom_idx, :].numpy()
            
            visu.plot_shift(embedding_matrix_1=previous_embeddings[population, cognom_idx,:],
                            embedding_matrix_2= cognom_embeddings[population],
                            fig_path=os.path.join(base_save_plot, "cognom_1", f"shift_{random_surname}.jpg"))
        except:
            print("Visualization error in shif plot")

        visu.plot_embedding_distribution(cognom_embeddings, os.path.join(base_save_plot, "cognom_1", f"distribution.jpg"))
   
        ## & Second surname shift plot
        #try:
        population = graph["cognom_2"].map_attribute_index[random_second_surname]
        cognom2_idx = list(graph.map_attribute_nodes.values()).index("cognom_2")
        cognom2_embeddings = attribute_embeddings[:, cognom2_idx, :].numpy()
        
        visu.plot_shift(embedding_matrix_1=previous_embeddings[population, cognom2_idx,:],
                        embedding_matrix_2 = cognom2_embeddings[population],
                        fig_path=os.path.join(base_save_plot, "cognom_2", f"shift_{random_second_surname}.jpg"))
        #except:
        #    print("Visualization error in shif plot")
        
        visu.plot_embedding_distribution(cognom2_embeddings, os.path.join(base_save_plot, "cognom_2", f"distribution.jpg"))


    if record_link == True:
        print("Evaluating Inter Attribute Metric Space")
        knn_metr = utils.evaluate_attribute_metric_space(attribute_embeddings.numpy(), plot_path=os.path.join(base_save_plot, f"inter_attr_distribution.jpg"))
        metrics.update(knn_metr)

    print("Evaluating the cluster distances")
    computed_freq_name = {name: len(values) for name, values in graph["nom"].map_attribute_index.items() if len(values) > 10}
    computed_freq_surname = {name: len(values) for name, values in graph["cognom_1"].map_attribute_index.items() if len(values) > 10}
    computed_freq_ssurname = {name: len(values) for name, values in graph["cognom_2"].map_attribute_index.items() if len(values) > 10}
    
    name_freq  = sorted(list(computed_freq_name.items()), key= lambda k: (k[1]), reverse=True)
    surname_freq  = sorted(list(computed_freq_surname.items()), key= lambda k: (k[1]), reverse=True)
    second_surnname_freq  = sorted(list(computed_freq_ssurname.items()), key= lambda k: (k[1]), reverse=True)
    
    most_common_names, less_common_names = name_freq[:10], name_freq[-10:]
    most_common_surnames, less_common_surnames = surname_freq[:10], surname_freq[-10:]
    most_common_ssurnames, less_common_ssurnames = second_surnname_freq[:10], second_surnname_freq[-10:]
    
    ## distributions most and less common names
    kepp_indexes_name = [(label, graph["nom"].map_attribute_index[label]) for label, _ in most_common_names + less_common_names  ]
    visu.plot_violin_plot_from_freq_attribute_distances(specific_attribute_embeddings=attribute_embeddings[:,0,:], dic_realtion_names=kepp_indexes_name,
                                                        file_path=os.path.join(base_save_plot, "nom", f"common_distances_distribution.jpg"))
    ## distributions most and less common surnames
    kepp_indexes_surname = [(label, graph["cognom_1"].map_attribute_index[label]) for label, _ in most_common_surnames + less_common_surnames  ]
    visu.plot_violin_plot_from_freq_attribute_distances(specific_attribute_embeddings=attribute_embeddings[:,1,:], dic_realtion_names=kepp_indexes_surname,
                                                        file_path=os.path.join(base_save_plot, "cognom_1", f"common_distances_distribution.jpg"))
    
    ## distributions most and less common second surnames
    kepp_indexes_ssurname = [(label, graph["cognom_2"].map_attribute_index[label]) for label, _ in most_common_ssurnames + less_common_ssurnames  ]
    visu.plot_violin_plot_from_freq_attribute_distances(specific_attribute_embeddings=attribute_embeddings[:,2,:], dic_realtion_names=kepp_indexes_ssurname,
                                                        file_path=os.path.join(base_save_plot, "cognom_2", f"common_distances_distribution.jpg"))

    if record_link == True:
        print("Evaluating the record linkage Task")

        candidate_pairs = core_graph[("individual", "similar", "individual")].negative_sampling
        true_pairs = core_graph[("individual", "similar", "individual")].edge_index
        
        # extract the population from the last time
        final_time_gap_population = list(core_graph.epoch_populations[-2]) + list(core_graph.epoch_populations[-1]) 
        final_time_gap_population = torch.tensor(final_time_gap_population).type(torch.int64)
        
        ## Extract candidate pairs 
        #specific_subgraph_candidate, _ = tutils.subgraph(subset=final_time_gap_population, edge_index=candidate_pairs[:2,:].type(torch.int64))
        ## Extract true pairs for the last epochs
        mask = torch.isin(true_pairs[:2, :], final_time_gap_population).all(dim=0)
        mask_candidate = torch.isin(candidate_pairs[:2,:], final_time_gap_population).all(dim=0)
        
        ## /& This one is used to get the final numbers ( This is train true pairs)
        specific_true_pairs_subgraph = true_pairs[:, mask]
        specific_subgraph_candidate = candidate_pairs[:, mask_candidate]
        
        ### Fins aquí tot bé
        
        ## & TEST SAME AS
        X_test_indexes = torch.cat((specific_true_pairs_subgraph, specific_subgraph_candidate), dim=1).type(torch.int32).numpy()
        X_test = (attribute_embeddings[X_test_indexes[0], :] - attribute_embeddings[X_test_indexes[1],:]).pow(2).sum(-1).sqrt()
        y_test = X_test_indexes[-1]
        
        
        ## ^ TRAIN EXTRACTION 
        earlies_time_populations = []
        for pop in core_graph.epoch_populations[:-1]:
            earlies_time_populations += list(pop)
            
            
        earlies_time_populations = torch.tensor(earlies_time_populations).type(torch.int64)

        mask = torch.isin(true_pairs[:2, :], earlies_time_populations).all(dim=0)
        mask_candidate_train = torch.isin(candidate_pairs[:2,:], earlies_time_populations).all(dim=0)
        
        specific_true_pairs_subgraph_train = true_pairs[:, mask]
        specific_subgraph_candidate_train = candidate_pairs[:, mask_candidate_train]

        X_train_indexes = torch.cat((specific_true_pairs_subgraph_train, specific_subgraph_candidate_train), dim=1).numpy()
        X_train = (attribute_embeddings[X_train_indexes[0], :] - attribute_embeddings[X_train_indexes[1],:]).pow(2).sum(-1).sqrt()

        y_train = X_train_indexes[-1] #torch.cat((torch.ones((specific_true_pairs_subgraph_train.shape[1])), torch.zeros((specific_subgraph_candidate_train.shape[1]))), dim=0).numpy()

        metrics_rl = rl.record_linkage_with_logistic_regression(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, candidate_sa=X_test_indexes,
                                                                                                                        random_state=0, penalty="l2", solver="newton-cholesky", class_weight="balanced", n_jobs=8)
        
        metrics["Record Linkage"] = metrics_rl
        
        print(metrics)
        
        print(F"SAVING ALL THE RESULTS IN: {os.path.join(base_save_metrics, 'results.json')}")
        with open(os.path.join(base_save_metrics, "results.json"), "w") as file:
            json.dump(metrics, file)
            

    return metrics
"""



"""
        #if w < self.general_width:
        #    image_line = transforms.resize(img=image_line, size=(self.general_height, self.general_width))
        #else:
        #    image_line = image_line[:, :, :self.general_width]
        #    image_line = transforms.resize(img=image_line, size=(self.general_height , self.general_width))    

        #image_line = transforms.normalize(image_line, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))   
        #image_line = self._transforms(image_line)

        #page_cutted = torch.from_numpy(page_cutted).permute(2, 0, 1)     
        #page_cutted = transforms.normalize(page_cutted, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ## Extract the ocr

"""

"""
    ## ^ Model 

    H = image_dataset.general_height 
    W = image_dataset.general_width // 16

    CFG_MODELS.edge_visual_encoder.input_atention_mechanism = H*W

    model = MMGCM(visual_encoder=VE.LineFeatureExtractor, gnn_encoder=gnn.AttributeGNN, edge_encoder=EVE.EdgeAttFeatureExtractor, cfg=CFG_MODELS).to(device)
    
    if CFG_MODELS.finetune is True:
        model_name = f"./checkpoints/{checkpoint_name}.pt"
        
        model.load_state_dict(torch.load(model_name))
        print("Model Loaded Succesfully Starting Finetunning")
    
    optimizer = hydra.utils.instantiate(CFG_SETUP.optimizer, params=model.parameters())

    if cfg.verbose == True:
        print("Configuration of the Models: ")
        bprint(dict(CFG_MODELS))

    if cfg.verbose == True:
        print("Inizialization Done without problemas")
        print("Starting the training proces with the following configuration: ")
        bprint(dict(CFG_SETUP.configuration)) 
    
    ## ^  

    optimal_loss = 10000
    
    nodes_to_compute_loss = CFG_SETUP.configuration.compute_loss_on
    core_graph = graphset._graph

    if CFG_MODELS.add_language is True:
        
        filepath  = f"embeddings/language_embeddings_{number_volum_years}_{embedding_size}_entities_{len(CFG_SETUP.configuration.compute_loss_on)}.pkl"
        
        if os.path.exists(filepath):
            language_embeddings = utils.read_pickle(filepath)
                
        else:
            language_embeddings = utils.extract_language_embeddings(list(image_dataset._map_ocr.keys()), embedding_size=embedding_size, filepath=filepath)
    
        print("LANGUAGE EMBEDDINGS EXTRACTED SUCCESFULLY")
    
        core_graph.x_language = language_embeddings

        print(image_dataset._map_ocr)


    core_graph.epoch_populations = image_dataset._population_per_volume
    
    
    
    criterion = losses.TripletMarginLoss(margin=0.2,
                    swap=False,
                    smooth_loss=False,
                    triplets_per_anchor="all")

"""



"""
            edge_similar_names = [(attribute, "similar", attribute) for attribute in entities]
            
            similar_populations = [
                subgraph[edge].flatten().unique().to(device) for edge in edge_similar_names
            ]
            labels_list = [torch.isin(population, sp).to(torch.int32) for sp in similar_populations]
            labels = torch.stack(labels_list, dim=0)  # Stack into a batch tensor
            
            individual_mask = torch.tensor([attribute == "individual" for attribute in entities])
            embeddings = torch.where(individual_mask.unsqueeze(-1).unsqueeze(-1),
                                    individual_embeddings.unsqueeze(0),  # Broadcasting for individuals
                                    attribute_representation.permute(1, 0, 2))
            
            loss = criterion(embeddings, labels)
"""

"""
def batch_step(loader, 
               graph:Type[HeteroData], 
               model: Type[nn.Module],
               optimizer: Type[torch.optim.Adam],
               criterion: Type[nn.Module],
               entities: Tuple[str, ...],
               scheduler,
               epoch: int,
               language_distillation:bool=False):

        

        model.train()
        
        epoch_train_loss = 0
        epoch_entities_losses = {key+"_train_loss":0 for key in entities}

        if language_distillation: 
            x_ocr = torch.from_numpy(np.array(graph.x_language)).to(device)

        
        for idx, dict_images in tqdm.tqdm(enumerate(loader), ascii=True):
            optimizer.zero_grad()

            images = dict_images["image_lines"].to(device)
            #ocr_indexes = dict_images["ocrs"].to(device)
            population = dict_images["population"].to(device)

            attribute_representation, individual_embeddings = model(x=images)#.encode_attribute_information(image_features=image_features, edge_attributes=edge_features) #message_passing
            
                
            ### *Task loss
            loss = 0
            batch_entites_loss = copy.copy(epoch_entities_losses)
            edges_to_keep = [(attribute, "similar", attribute) for attribute in entities]
            subgraph = utils.extract_subgraph(graph=graph, population=population, edges_to_extract=edges_to_keep)

            
            for idx, attribute in enumerate(entities):
                

                edge_similar_name = (attribute, "similar", attribute)
                similar_population = subgraph[edge_similar_name].flatten().unique().to(device)                
                
                labels = torch.isin(population, similar_population).type(torch.int32)
                # Select embeddings based on the attribute
                if attribute == "individual":
                    embeddings = individual_embeddings
                else:
                    embeddings = attribute_representation[:, idx, :]

                    
                # Only compute language distillation if enabled
                if language_distillation:
                    ocr_indexes = dict_images["ocrs"].to(device)
                    selected_language_embeddings_idx = ocr_indexes[:, idx].to(torch.int32)
                    language_embeddings = x_ocr[selected_language_embeddings_idx, :]
                    
                    # Concatenate embeddings and labels in a single operation
                    labels = torch.cat((labels, labels), dim=0)
                    embeddings = torch.cat((embeddings, language_embeddings), dim=0)

                loss_attribute = criterion(embeddings, labels)
                loss += (loss_attribute) # + contrastive_loss_attribute) #distilattion_loss
                
                batch_entites_loss[attribute+"_train_loss"] += loss_attribute

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            
        
        epoch_train_loss += loss
        epoch_entities_losses.update(batch_entites_loss)
        epoch_entities_losses["train_loss"] = epoch_train_loss

        print(f"Epoch {epoch}: Loss: {epoch_train_loss}")

        wandb.log(epoch_entities_losses)
        
        
        return loss

"""