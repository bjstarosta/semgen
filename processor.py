# -*- coding: utf-8 -*-
"""SEMGen - Processor superclass.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging
import multiprocessing as mp

import utils


class Processor(object):
    """Abstract parent class for all the generator subclasses.

    Subclasses become iterators operating on a queue.

    Attributes:
        dim (tuple): Dimensions of the generated image in (width, height)
            format.
    """

    def __init__(self):
        self.dim = (400, 400)
        self._queue = []

    def __iter__(self):
        while len(self._queue) > 0:
            yield self.get()

    def __next__(self):
        if len(self._queue) > 0:
            return self.get()
        else:
            return

    def __enter__(self):
        # dummy for the "with" keyword so progress bars can be disabled easily
        return self

    def __exit__(self, type, value, traceback):
        pass

    def get(self, task):
        """Pop the first element off of the queue, process and return it.

        Returns:
            processor.ProcessorTask: Completed task object.

        """
        task.prepare()
        return self.process(task)

    def enqueue(self, n, files_out, files_in=[], params=[]):
        """Enqueue a sequence of tasks.

        The values of the other parameters will be distributed to individual
        ProcessorTask objects.

        Args:
            n (int): Number of tasks to enqueue.
            files_out (list): List of file paths to output results to.
            files_in (list): List of file paths to images to load for
                processing. Optional.
            params (list): List of image processing parameters. If empty
                or list is smaller than number of tasks to enqueue, the
                remaining parameters will be randomly generated.

        Returns:
            None

        """
        assert n <= len(files_out)
        self._queue = []
        for i in range(0, n):
            task = ProcessorTask(i, files_out[i])
            if len(files_in) > i:
                task.in_file = files_in[i]
            if len(params) > i:
                task.params = params[i]
                params_generated = False
            else:
                task.params = self.generate_params()
                params_generated = True

            logging.debug(
                "Setting task {0}: "
                "out_file='{1}', in_file='{2}', params_generated={3}",
                i, task.out_file, task.in_file, params_generated)
            self._queue.append(task)
        logging.debug(
            "Queue length: {0}. "
            "len(files_out)={1}, len(files_in)={2}, len(params)={3}",
            len(self._queue), len(files_out), len(files_in), len(params))

    def get_globals(self):
        """Return list of processor specific properties.

        Returns:
            list: List of processor specific properties.

        """
        d = {}
        n = ['_queue']
        for k in self.__dict__.keys():
            if k not in n and k[0] != '_':
                d[k] = self.__dict__[k]
        return d

    def generate_params(self):
        """Randomly generate processing parameters for a task.

        Should be implemented by subclasses.

        Should always use np.random.RandomState() instantiation when generating
        random numbers for reasons of thread safety.

        Returns:
            dict: A dictionary of variables to be used during image processing.
                Can be arbitrary with regards to structure, the reason for the
                separation is so that these parameters can be externally saved
                for later reproducibility.

        """
        raise NotImplementedError()

    def process(self, task):
        """Process passed task and return the same object containing a result.

        Should be implemented by subclasses.

        Args:
            task (processor.ProcessorTask): Task object to be processed.

        Returns:
            processor.ProcessorTask: Processed task object.

        """
        raise NotImplementedError()

    def process_all(self, n_proc=1):
        """Process all queued tasks in a non blocking way and return the result.

        Args:
            n_proc (int): Number of processes to spawn.

        Returns:
            list: List of ProcessorTask objects. Iterable before completion.

        """
        pool = mp.Pool(processes=n_proc)
        return pool.imap_unordered(
            func=self.get,
            iterable=self._queue
        )

    def _getpx(self, im, x, y):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        return im[y, x]

    def _setpx(self, im, x, y, c):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        im[y, x] = c

    def _slice(self, src, dst, x, y):
        """Safely slices two numpy arrays together.

        Truncates the source array as necessary to fit the destination array.
        X and Y coordinates can be negative numbers, and the source will be
        truncated accordingly.

        Args:
            src (ndarray): Source array to be sliced into the destination.
            dst (ndarray): Destination array.
            x (int): X-axis coordinate to paste the array at.
            y (int): y-axis coordinate to paste the array at.

        Returns:
            A numpy array the size of the destination array.

        """
        if (x < -src.shape[1] or y < -src.shape[0]
        or x > dst.shape[1] or y > dst.shape[0]):
            return dst

        if x < 0:
            src = src[:, x:]
            x = 0
        if y < 0:
            src = src[y:, :]
            y = 0
        if (x + src.shape[1]) > dst.shape[1]:
            d = (x + src.shape[1]) - dst.shape[1]
            src = src[:, :-d]
        if (y + src.shape[0]) > dst.shape[0]:
            d = (y + src.shape[0]) - dst.shape[0]
            src = src[:-d, :]

        dst[y:y + src.shape[0], x:x + src.shape[1]] = src
        return dst


class ProcessorTask(object):
    """Data structure used in passing data to the processor.

    Args:
        index (int): Index of the task in the queue.
        out_file (str): Location of output file.
        in_file (str): Location of input file. Optional.

    Attributes:
        params (dict): Dictionary of processor specific parameters.
        labels (list): List of labels.
        image (numpy.ndarray): Result image.
        completed (bool): Set to True if task has been processed,
            False otherwise.
        index
        out_file
        in_file

    """

    def __init__(self, index, out_file, in_file=None):
        self.index = index
        self.out_file = out_file
        self.in_file = in_file
        self.params = None
        self.labels = None
        self.image = None
        self.completed = False

    def prepare(self):
        """Load any necessary data and prepare the task for processing.

        Returns:
            None

        """
        if self.in_file is not None:
            self.image = utils.load_image(self.in_file)

    def complete(self, paramfile=None, labelfile=None, resize=None):
        """Run post processing routines on a task.

        Args:
            paramfile (params.ParamFile): Relevant parameter file handler.
            labelfile (labels.LabelFile): Relevant label file handler.

        Returns:
            None

        """
        if self.completed is True:
            return

        out_basename = os.path.basename(self.out_file)
        if paramfile is not None:
            if self.params is None:
                logging.debug(
                    "Task {0}: paramfile was passed but params property was "
                    "not set in the task object. Passing empty list.",
                    self.index)
                paramfile.append([], out_basename)
            else:
                paramfile.append(self.params, out_basename)

        if labelfile is not None:
            if self.labels is None:
                logging.debug(
                    "Task {0}: labelfile was passed but labels property was "
                    "not set in the task object. Passing empty list.",
                    self.index)
                labelfile.add_label([])
            else:
                labelfile.add_label(self.labels)
            labelfile.add_file(out_basename)

        if resize is not None:
            self.image = utils.resize_image(self.image, resize)

        utils.save_image(self.out_file, self.image)
        self.completed = True
