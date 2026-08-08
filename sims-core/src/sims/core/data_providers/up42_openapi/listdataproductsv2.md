---
updatedAt: 2026-05-27T08:26:46.000Z
---

Fetch the complete documentation index at: https://developer.up42.com/llms.txt. Use this file to discover all available pages before exploring further.

# Get data products

Get a list of data products. A data product is a type of imagery with a specific geometric and radiometric processing level and file format.


# OpenAPI definition

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "UP42 API",
    "contact": {
      "name": "Contact support",
      "email": "support@up42.com"
    },
    "license": {
      "name": "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License",
      "url": "http://creativecommons.org/licenses/by-nc-nd/4.0/"
    },
    "version": "1.2"
  },
  "servers": [
    {
      "url": "https://api.up42.com"
    }
  ],
  "security": [
    {
      "httpBearer": []
    }
  ],
  "tags": [
    {
      "name": "Glossary"
    }
  ],
  "paths": {
    "/v2/data-products": {
      "get": {
        "tags": [
          "Glossary"
        ],
        "summary": "Get data products",
        "description": "Get a list of data products. A data product is a type of imagery with a specific geometric and radiometric processing level and file format.\n",
        "operationId": "listDataProductsV2",
        "parameters": [
          {
            "$ref": "#/components/parameters/DataProductId"
          },
          {
            "$ref": "#/components/parameters/PageablePage"
          },
          {
            "$ref": "#/components/parameters/PageableSize"
          },
          {
            "$ref": "#/components/parameters/PageableSorting"
          }
        ],
        "responses": {
          "200": {
            "description": "OK",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/PageOfDataProducts"
                }
              }
            }
          }
        },
        "security": []
      }
    }
  },
  "components": {
    "schemas": {
      "Sort": {
        "type": "object",
        "properties": {
          "empty": {
            "type": "boolean",
            "example": false
          },
          "sorted": {
            "type": "boolean",
            "example": true
          },
          "unsorted": {
            "type": "boolean",
            "example": false
          }
        }
      },
      "PageableObject": {
        "type": "object",
        "properties": {
          "pageNumber": {
            "type": "integer",
            "format": "int32",
            "example": 0
          },
          "pageSize": {
            "type": "integer",
            "format": "int32",
            "example": 10
          },
          "sort": {
            "$ref": "#/components/schemas/Sort"
          },
          "offset": {
            "type": "integer",
            "format": "int64",
            "example": 0
          },
          "paged": {
            "type": "boolean",
            "example": true
          },
          "unpaged": {
            "type": "boolean",
            "example": false
          }
        }
      },
      "CollectionType": {
        "type": "string",
        "description": "The type of the collection.\n",
        "example": "ARCHIVE",
        "enum": [
          "ARCHIVE",
          "TASKING"
        ]
      },
      "Collection": {
        "required": [
          "description",
          "name",
          "title",
          "type"
        ],
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the collection.\n",
            "example": "vexcel-elevate-dtm-15cm"
          },
          "title": {
            "type": "string",
            "description": "The title of the collection.\n",
            "example": "Vexcel Elevate — DTM 15 cm"
          },
          "description": {
            "type": "string",
            "description": "The description of the collection.\n",
            "example": "A very high resolution 15 cm digital terrain model, acquired over areas all over the world. The collection contains data from 2019 to the present."
          },
          "type": {
            "$ref": "#/components/schemas/CollectionType"
          }
        },
        "description": "Information about the collection.\n"
      },
      "Provider": {
        "required": [
          "description",
          "name",
          "title"
        ],
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the provider.\n",
            "example": "vexcel-v2"
          },
          "title": {
            "type": "string",
            "description": "The title of the provider.\n",
            "example": "Vexcel"
          },
          "description": {
            "type": "string",
            "description": "The description of the provider.\n",
            "example": "Vexcel delivers geospatial data products with high accuracy, spatial resolution, and consistency. The Vexcel Data Program is the largest in the world capturing ultra-high-resolution imagery (at 7.5 to 15cm resolution) and related geospatial data in more than 30 countries, including the U.S., Canada, U.K., Western and Eastern Europe, Australia, New Zealand, and Japan."
          }
        }
      },
      "PageableResponse": {
        "type": "object",
        "properties": {
          "pageable": {
            "$ref": "#/components/schemas/PageableObject"
          },
          "last": {
            "type": "boolean",
            "example": false
          },
          "totalElements": {
            "type": "integer",
            "format": "int64",
            "example": 100
          },
          "totalPages": {
            "type": "integer",
            "format": "int32",
            "example": 20
          },
          "size": {
            "type": "integer",
            "format": "int32",
            "example": 10
          },
          "number": {
            "type": "integer",
            "format": "int32",
            "example": 0
          },
          "sort": {
            "$ref": "#/components/schemas/Sort"
          },
          "numberOfElements": {
            "type": "integer",
            "format": "int32",
            "example": 10
          },
          "first": {
            "type": "boolean",
            "example": true
          },
          "empty": {
            "type": "boolean",
            "example": false
          }
        }
      },
      "ResolutionValue": {
        "required": [
          "minimum"
        ],
        "type": "object",
        "properties": {
          "minimum": {
            "type": "number",
            "description": "The best possible resolution available for the collection, in meters.\n",
            "example": 0.15
          },
          "maximum": {
            "type": "number",
            "description": "The least detailed resolution available for the collection, in meters. The parameter is omitted if the collection has only one resolution value, specified in `minimum`.\n",
            "nullable": true,
            "example": 0.2
          }
        },
        "description": "The level of detail achievable for the collection.\n"
      },
      "ProductType": {
        "type": "string",
        "description": "The type of product the collection delivers.\n",
        "example": "ELEVATION",
        "enum": [
          "OPTICAL",
          "SAR",
          "ELEVATION"
        ]
      },
      "ProviderWithRoles": {
        "allOf": [
          {
            "$ref": "#/components/schemas/ProviderRolesArray"
          },
          {
            "$ref": "#/components/schemas/Provider"
          }
        ]
      },
      "ProviderRolesArray": {
        "required": [
          "roles"
        ],
        "type": "object",
        "properties": {
          "roles": {
            "type": "array",
            "description": "Provider roles:\n  - `PRODUCER`: a producer is a provider that initially acquired and processed the source data. Data acquired by a producer can be distributed to various hosts.\n  - `HOST`: a host is a provider that offers access to data acquired by a producer.\n",
            "example": [
              "HOST",
              "PRODUCER"
            ],
            "items": {
              "$ref": "#/components/schemas/ProviderRole"
            }
          }
        }
      },
      "ProviderRole": {
        "type": "string",
        "enum": [
          "PRODUCER",
          "HOST"
        ]
      },
      "CollectionExtended": {
        "allOf": [
          {
            "$ref": "#/components/schemas/Collection"
          },
          {
            "$ref": "#/components/schemas/CollectionInfoExtension"
          }
        ]
      },
      "CollectionInfoExtension": {
        "required": [
          "integrations"
        ],
        "type": "object",
        "properties": {
          "metadata": {
            "$ref": "#/components/schemas/CollectionMetadata"
          },
          "integrations": {
            "$ref": "#/components/schemas/CollectionIntegrationArray"
          }
        }
      },
      "CollectionIntegrationArray": {
        "type": "array",
        "description": "[Integration values](https://docs.up42.com/developers/api-glossary#integrations) that indicate mandatory ordering steps, available actions, and mandatory or optional operations for a given collection.\n",
        "example": [
          "SAMPLE_DATA_AVAILABLE",
          "QUICKLOOK_AVAILABLE",
          "SEARCH_AVAILABLE",
          "PRICE_ESTIMATION_AVAILABLE",
          "THUMBNAIL_AVAILABLE"
        ],
        "items": {
          "$ref": "#/components/schemas/CollectionIntegration"
        }
      },
      "CollectionIntegration": {
        "type": "string",
        "enum": [
          "ACCESS_APPROVAL_REQUIRED",
          "SAMPLE_DATA_AVAILABLE",
          "MANUAL_REQUEST_REQUIRED",
          "FEASIBILITY_STUDY_REQUIRED",
          "FEASIBILITY_STUDY_MAY_BE_REQUIRED",
          "QUOTATION_REQUIRED",
          "PRICE_ESTIMATION_AVAILABLE",
          "SEARCH_AVAILABLE",
          "THUMBNAIL_AVAILABLE",
          "QUICKLOOK_AVAILABLE",
          "TASKING_OPPORTUNITIES_AVAILABLE"
        ]
      },
      "CollectionMetadata": {
        "type": "object",
        "properties": {
          "productType": {
            "$ref": "#/components/schemas/ProductType"
          },
          "resolutionClass": {
            "$ref": "#/components/schemas/ResolutionClass"
          },
          "resolutionValue": {
            "$ref": "#/components/schemas/ResolutionValue"
          },
          "priority": {
            "$ref": "#/components/schemas/PriorityArray"
          },
          "acquisitionMode": {
            "$ref": "#/components/schemas/AcquisitionModeArray"
          }
        },
        "description": "The collection metadata."
      },
      "ResolutionClass": {
        "type": "string",
        "description": "The spatial resolution class.\n",
        "example": "VERY_HIGH",
        "enum": [
          "VERY_HIGH",
          "HIGH",
          "MEDIUM",
          "LOW"
        ]
      },
      "DataProduct": {
        "required": [
          "description",
          "name",
          "title"
        ],
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "description": "The data product ID.\n",
            "format": "uuid",
            "example": "d28f2b14-b8c2-4b3c-b184-11334933eeb2"
          },
          "name": {
            "type": "string",
            "description": "The data product name.\n",
            "example": "vexcel-elevate-dtm-15cm"
          },
          "title": {
            "type": "string",
            "description": "The title of the data product.\n",
            "example": "Vexcel Elevate — DTM 15 cm"
          },
          "description": {
            "type": "string",
            "description": "The description of the data product.\n",
            "example": "A digital terrain model with a resolution of 15–20 cm."
          },
          "eulaId": {
            "type": "string",
            "description": "The EULA ID.\n",
            "format": "uuid",
            "example": "f79efe14-bc8d-4892-bd9d-7565cb3620a5"
          }
        }
      },
      "CollectionAggregatedWithProviders": {
        "allOf": [
          {
            "$ref": "#/components/schemas/CollectionExtended"
          },
          {
            "required": [
              "providers"
            ],
            "type": "object",
            "properties": {
              "providers": {
                "type": "array",
                "items": {
                  "$ref": "#/components/schemas/ProviderWithRoles"
                }
              }
            }
          }
        ]
      },
      "PageOfDataProducts": {
        "allOf": [
          {
            "$ref": "#/components/schemas/PageableResponse"
          },
          {
            "type": "object",
            "properties": {
              "content": {
                "type": "array",
                "items": {
                  "$ref": "#/components/schemas/DataProductAggregate"
                }
              }
            }
          }
        ]
      },
      "DataProductAggregate": {
        "type": "object",
        "allOf": [
          {
            "$ref": "#/components/schemas/DataProduct"
          },
          {
            "type": "object",
            "properties": {
              "collection": {
                "$ref": "#/components/schemas/CollectionAggregatedWithProviders"
              }
            }
          }
        ]
      },
      "PriorityArray": {
        "type": "array",
        "description": "[Priority tiers](https://docs.up42.com/data/reference/priority) available for a tasking collection.\n",
        "example": [
          "STANDARD"
        ],
        "items": {
          "$ref": "#/components/schemas/Priority"
        }
      },
      "Priority": {
        "type": "string",
        "enum": [
          "STANDARD",
          "HIGH",
          "RUSH"
        ]
      },
      "AcquisitionModeArray": {
        "type": "array",
        "description": "[Acquisition modes](https://docs.up42.com/data/reference/acquisition-modes) available for a collection.\n",
        "example": [
          "MONO"
        ],
        "items": {
          "$ref": "#/components/schemas/AcquisitionMode"
        }
      },
      "AcquisitionMode": {
        "type": "string",
        "enum": [
          "MONO",
          "STEREO",
          "TRISTEREO",
          "SPOT",
          "SPOT_DWELL",
          "SPOT_DWELL_FINE",
          "SPOT_DWELL_PRECISE",
          "SPOT_FINE",
          "SPOT_EXTENDED",
          "SPOT_ENHANCED",
          "SPOT_ULTRA",
          "SPOT_HIGH_RES",
          "SPOT_HIGH_RES_300",
          "SITE",
          "STRIP",
          "STRIP_ENHANCED",
          "SCAN",
          "SCAN_WIDE",
          "SCAN_ENHANCED",
          "STEREO_BURST",
          "STEREO_AREA_COVERAGE",
          "STEREO_SINGLE_PASS",
          "STEREO_MULTI_PASS"
        ]
      }
    },
    "parameters": {
      "PageableSize": {
        "name": "size",
        "in": "query",
        "description": "The number of results on a result page.\n",
        "required": false,
        "style": "form",
        "explode": true,
        "schema": {
          "type": "integer",
          "example": 10,
          "default": 20
        }
      },
      "PageablePage": {
        "name": "page",
        "in": "query",
        "description": "The result page number. To get the first page, set the parameter to `0`.\n",
        "required": false,
        "style": "form",
        "explode": true,
        "schema": {
          "type": "integer",
          "example": 1,
          "default": 0
        }
      },
      "PageableSorting": {
        "name": "sort",
        "in": "query",
        "description": "The results sorting method that arranges elements in ascending or descending order based on a chosen field. The format is `<field name>,<asc or desc>`.\n",
        "required": false,
        "style": "form",
        "explode": true,
        "schema": {
          "type": "string",
          "example": "id,asc",
          "default": "createdAt,desc"
        }
      },
      "DataProductId": {
        "name": "dataProductId",
        "in": "query",
        "description": "The data product ID.\n",
        "required": false,
        "style": "form",
        "explode": true,
        "schema": {
          "type": "string",
          "format": "uuid",
          "example": "8fbccd60-c828-4d8c-a4ef-3c2730d9d6bc"
        }
      }
    },
    "securitySchemes": {
      "httpBearer": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
      }
    }
  }
}
```