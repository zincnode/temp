# Copyright 2019 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import six

from dashboard.api import api_request_handler
from dashboard.common import namespaced_stored_object
from dashboard import revision_info_client

ALLOWLIST = [
    revision_info_client.REVISION_INFO_KEY,
]

from flask import request

def _CheckUser():
  pass

@api_request_handler.RequestHandlerDecoratorFactory(_CheckUser)
def ConfigHandlerPost():
  logging.debug('crbug/1298177 - /api/config handler triggered')
  key = request.values.get('key')
  if key not in ALLOWLIST:
    return None
  return namespaced_stored_object.Get(key)

if six.PY2:
  # pylint: disable=abstract-method
  class ConfigHandler(api_request_handler.ApiRequestHandler):

    def _CheckUser(self):
      pass

    def Post(self, *args, **kwargs):
      del args, kwargs  # Unused.
      key = self.request.get('key')
      if key not in ALLOWLIST:
        return None
      return namespaced_stored_object.Get(key)
