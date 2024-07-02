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