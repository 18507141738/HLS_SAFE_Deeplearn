# -*- coding: utf-8 -*-


class AlgorithmBase(object):
    def __init__(self, save_flag):
        self.save_flag = save_flag
        self.state = True
        self.output_img = None

    def loop(self):
        """
        keep algorithm running
        keep updating self.output_img
        """
        raise NotImplementedError()

    def set_state(self, state):
        """
        Set status for working thread.
        """
        self.state = state

    def get_output(self):
        return self.output_img
