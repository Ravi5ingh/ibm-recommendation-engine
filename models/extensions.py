class dictionary(dict):
    """
    This is basically a more awesome version of the python dictionary
    """

    def key_at(self, key_index):
        """
        Get the key at a given index
        :param key_index: The key index
        :return: The key at the given index
        """

        return list(self.keys())[key_index]

    def value_at(self, value_index):
        """
        Get the value at the given index
        :param value_index: The value index
        :return: The value at the given index
        """

        return list(self.values())[value_index]

    def get_sorted(self, by_val=True, ascending=True):
        """
        Sort the dictionary and return the sorted version
        :param by_val: Whether or not to sort by value (Default: True)
        :param ascending: Whether or not the sort in ascending order (Default: True)
        :return: The sorted dictionary
        """

        return dictionary((sorted(self.items(), key=lambda x: x[1 if by_val else 0], reverse=not ascending)))