# Copyright 2018 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging

from dashboard.api import api_request_handler
# Module imported for its side effects, to register static report templates.
import dashboard.common.system_health_report  # pylint: disable=unused-import
from dashboard.models import report_template


# pylint: disable=abstract-method
class ReportNamesHandler(api_request_handler.ApiRequestHandler):

  def _CheckUser(self):
    pass

  def Post(self, *args, **kwargs):
    logging.debug('crbug/1298177 - /api/report_names handler triggered')
    del args, kwargs  # Unused.
    return report_template.List()
