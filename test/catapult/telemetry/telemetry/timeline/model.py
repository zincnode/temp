# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
"""A container for timeline-based events and traces and can handle importing
raw event data from different sources. This model closely resembles that in the
trace_viewer project:
https://code.google.com/p/trace-viewer/
"""

from __future__ import absolute_import
import logging
import itertools

from operator import attrgetter

import six

from telemetry.timeline import bounds
from telemetry.timeline import event_container
from telemetry.timeline import process as process_module
from telemetry.timeline import trace_event_importer

from tracing.trace_data import trace_data as trace_data_module


class MarkerMismatchError(Exception):
  def __init__(self):
    super().__init__(
        'Number or order of timeline markers does not match provided labels')


class MarkerOverlapError(Exception):
  def __init__(self):
    super().__init__(
        'Overlapping timeline markers found')


def _ExtractTraceData(container):
  if isinstance(container, trace_data_module.TraceDataBuilder):
    try:
      return container.AsData()
    finally:
      container.CleanUpTraceData()
  else:
    return container


class TimelineModel(event_container.TimelineEventContainer):
  def __init__(self, trace_data=None, shift_world_to_zero=True):
    """Initializes a TimelineModel.

    This class is deprecated, no new clients should use it.

    Args:
        trace_data: Either a TraceDataBuilder object, or a readable trace data
            value as generated by its AsData() method. When passing a builder,
            the data will be extracted from it, and the resources used by the
            builder cleaned up. This is to support legacy clients which were
            not aware of the responsibility to clean up trace data.
        shift_world_to_zero: If true, the events will be shifted such that the
            first event starts at time 0.
    """
    super().__init__(name='TimelineModel', parent=None)
    self._bounds = bounds.Bounds()
    self._thread_time_bounds = {}
    self._processes = {}
    self._browser_process = None
    self._gpu_process = None
    self._surface_flinger_process = None
    self._frozen = False
    self.import_errors = []
    self.metadata = []
    self.flow_events = []
    self._global_memory_dumps = None
    if trace_data is not None:
      self._ImportTraces(_ExtractTraceData(trace_data),
                         shift_world_to_zero=shift_world_to_zero)

  def SetGlobalMemoryDumps(self, global_memory_dumps):
    """Populates the model with a sequence of GlobalMemoryDump objects."""
    assert not self._frozen and self._global_memory_dumps is None
    # Keep dumps sorted in chronological order.
    self._global_memory_dumps = tuple(sorted(global_memory_dumps,
                                             key=lambda dump: dump.start))

  def IterGlobalMemoryDumps(self):
    """Iterate over the memory dump events of this model."""
    return iter(self._global_memory_dumps or [])

  def IterChildContainers(self):
    for process in six.itervalues(self._processes):
      yield process

  def GetAllProcesses(self):
    return list(self._processes.values())

  def GetAllThreads(self):
    threads = []
    for process in self._processes.values():
      threads.extend(list(process.threads.values()))
    return threads

  @property
  def bounds(self):
    return self._bounds

  @property
  def processes(self):
    return self._processes

  @property
  def browser_process(self):
    return self._browser_process

  @browser_process.setter
  def browser_process(self, browser_process):
    self._browser_process = browser_process

  @property
  def gpu_process(self):
    return self._gpu_process

  @gpu_process.setter
  def gpu_process(self, gpu_process):
    self._gpu_process = gpu_process

  @property
  def surface_flinger_process(self):
    return self._surface_flinger_process

  @surface_flinger_process.setter
  def surface_flinger_process(self, surface_flinger_process):
    self._surface_flinger_process = surface_flinger_process

  def _ImportTraces(self, trace_data, shift_world_to_zero=True):
    """Populates the model with the provided trace data.

    Passing shift_world_to_zero=True causes the events to be shifted such that
    the first event starts at time 0.
    """
    if self._frozen:
      raise Exception("Cannot add events once trace is imported")

    importers = self._CreateImporters(trace_data)

    for importer in importers:
      # TODO: catch exceptions here and add it to error list
      importer.ImportEvents()
    self.FinalizeImport(shift_world_to_zero, importers)

  def FinalizeImport(self, shift_world_to_zero=False, importers=None):
    if importers is None:
      importers = []
    self.UpdateBounds()
    if not self.bounds.is_empty:
      for process in six.itervalues(self._processes):
        process.AutoCloseOpenSlices(self.bounds.max,
                                    self._thread_time_bounds)

    for importer in importers:
      importer.FinalizeImport()

    for process in six.itervalues(self.processes):
      process.FinalizeImport()

    if shift_world_to_zero:
      self.ShiftWorldToZero()
    self.UpdateBounds()

    # Because of FinalizeImport, it would probably be a good idea
    # to prevent the timeline from from being modified.
    self._frozen = True

  def ShiftWorldToZero(self):
    self.UpdateBounds()
    if self._bounds.is_empty:
      return
    shift_amount = self._bounds.min
    for event in self.IterAllEvents():
      event.start -= shift_amount

  def UpdateBounds(self):
    self._bounds.Reset()
    for event in self.IterAllEvents():
      self._bounds.AddValue(event.start)
      self._bounds.AddValue(event.end)

    self._thread_time_bounds = {}
    for thread in self.GetAllThreads():
      self._thread_time_bounds[thread] = bounds.Bounds()
      for event in thread.IterEventsInThisContainer(
          event_type_predicate=lambda t: True,
          event_predicate=lambda e: True):
        if event.thread_start is not None:
          self._thread_time_bounds[thread].AddValue(event.thread_start)
        if event.thread_end is not None:
          self._thread_time_bounds[thread].AddValue(event.thread_end)

  def GetOrCreateProcess(self, pid):
    if pid not in self._processes:
      assert not self._frozen
      self._processes[pid] = process_module.Process(self, pid)
    return self._processes[pid]

  def FindTimelineMarkers(self, timeline_marker_names):
    """Find the timeline events with the given names.

    If the number and order of events found does not match the names,
    raise an error.
    """
    # Make sure names are in a list and remove all None names
    if isinstance(timeline_marker_names, six.string_types):
      timeline_marker_names = [timeline_marker_names]
    names = [x for x in timeline_marker_names if x is not None]

    # Gather all events that match the names and sort them.
    events = list(self.IterTimelineMarkers(names))
    events.sort(key=attrgetter('start'))

    # Check if the number and order of events matches the provided names,
    # and that the events don't overlap.
    if len(events) != len(names):
      raise MarkerMismatchError()
    for (i, event) in enumerate(events):
      if event.name != names[i]:
        raise MarkerMismatchError()
    for event_i, event_j in itertools.combinations(events, 2):
      if event_j.start < event_i.start + event_i.duration:
        raise MarkerOverlapError()

    return events

  def GetFirstRendererProcess(self, tab_id):
    """Find the process for the first renderer thread in the model."""
    return self.GetFirstRendererThread(tab_id).parent

  def GetFirstRendererThread(self, tab_id):
    """Find the first renderer thread in the model for an expected tab

    This normally corresponds to the foreground tab at the time when the trace
    was collected.

    Args:
      tab_id: To make sure clients have gotten the correct renderer for an
          expected tab, they must also pass its id to verify.

    Raises an error if the thread cannot be found.
    """
    markers = self.FindTimelineMarkers('first-renderer-thread')
    assert len(markers) == 1
    renderer_thread = markers[0].start_thread
    assert renderer_thread == markers[0].end_thread
    verifiers = list(renderer_thread.IterTimelineMarkers(tab_id))
    assert len(verifiers) == 1, 'Renderer thread does not have expected tab id'
    return renderer_thread

  def _CreateImporters(self, trace_data):
    # Only TraceEventTimelineImporter (for CHROME_TRACE_PART) is supported.
    # Do not add any new importers to this deprecated TimelineModel class.
    importer_cls = trace_event_importer.TraceEventTimelineImporter
    importer_part = importer_cls.GetSupportedPart()

    importers = []
    if trace_data.HasTracesFor(importer_part):
      importers.append(importer_cls(self, trace_data))
    else:
      logging.warning('No traces found for %r', importer_part)

    return importers
