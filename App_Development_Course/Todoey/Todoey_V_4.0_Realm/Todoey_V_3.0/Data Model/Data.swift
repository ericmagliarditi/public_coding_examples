//
//  Data.swift
//  Todoey_V_3.0
//
//  Created by Eric Magliarditi on 6/23/19.
//  Copyright © 2019 Eric Magliarditi. All rights reserved.
//

import Foundation
import RealmSwift

class Data: Object {
    
    /**
     Inherit this object class in our new class that allows us to persist the data class. We will declare the properties in this class
     Since we are using realm - we need to mark our attributes with the key word dynamic (see below)
     Dynamic means - declaration modifier and tells the runtime to use dynamic dispatch over static dispatch - allows the property or attribute to be monitored for change while app is running - so if user changes value of attribute, realm will dynamically update this in the database
        -dynamic comes from objective c so need to add a decorator to ensure complier sees this
    */
    
    @objc dynamic var name: String = ""
    
    
    
    
    
    
    
    
    
    
    
    
    
}
